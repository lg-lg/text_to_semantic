from pathlib import Path
from tqdm import tqdm
import argparse
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from optimizer import get_optimizer, ScheduledOptim
from data import TextSemanticDataset
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils import cycle, load_text_to_semantic_model, sequence_mask, compute_loss, accum_log, exists, \
    get_config_from_file, calculate_model_params
import numpy as np
import datetime


class TextToSemanticTrainer(nn.Module):
    def __init__(self,
                 transformer,
                 config,
                 log_type='tensorboard',
                 logging_dir='./logs',
                 accelerate_kwargs: dict = dict(),
                 ):
        super().__init__()
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs], log_with=log_type, project_dir=logging_dir,
                                       gradient_accumulation_steps=config.train.grad_accum_every,
                                       **accelerate_kwargs)
        result_folder = os.path.join('checkpoints', config.hparams.name)
        self.accelerator.print(result_folder)
        self.hp = config.train
        self.transformer = transformer
        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = self.hp.num_train_steps
        self.batch_size = self.hp.batch_size
        self.grad_accum_every = self.hp.grad_accum_every
        self.num_workers = self.hp.num_workers
        self.max_text_length = self.hp.max_text_length
        self.max_history_semantic_length = self.hp.max_history_semantic_length

        # optimizers
        # 大约0.036 对应768
        self.optim = get_optimizer(self.transformer.parameters(), lr=self.hp.lr * np.sqrt(
            self.hp.grad_accum_every * self.batch_size * self.accelerator.num_processes), wd=self.hp.wd)
        self.scheduler_func = ScheduledOptim(warmup_steps=self.hp.warmup_steps, num_gpu=self.accelerator.num_processes)
        self.scheduler = LambdaLR(self.optim, self.scheduler_func.get_lr_scale)
        # max grad norm

        self.max_grad_norm = self.hp.max_grad_norm

        # create dataset
        self.ds = TextSemanticDataset(
            meta_file=self.hp.train_data_folder,
            max_text_length=self.hp.max_text_length,
            max_history_semantic_length=self.hp.max_history_semantic_length,
            max_semantic_length=self.hp.max_semantic_length,
            semantic_pad=self.hp.semantic_pad,
            text_pad=self.hp.text_pad,
            semantic_infer=self.hp.semantic_infer,
            text_offset=self.hp.text_offset,
        )
        self.valid_ds = TextSemanticDataset(
            meta_file=self.hp.test_data_folder,
            max_text_length=self.hp.max_text_length,
            max_history_semantic_length=self.hp.max_history_semantic_length,
            max_semantic_length=self.hp.max_semantic_length,
            semantic_pad=self.hp.semantic_pad,
            text_pad=self.hp.text_pad,
            semantic_infer=self.hp.semantic_infer,
            text_offset=self.hp.text_offset,
        )

        # dataloader
        self.dl = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True,
                             num_workers=self.num_workers,
                             collate_fn=self.ds.collate_fn)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers,
                                   collate_fn=self.valid_ds.collate_fn)

        # prepare with accelerator
        (
            self.transformer,
            self.optim,
            self.dl,
            self.scheduler
        ) = self.accelerator.prepare(
            self.transformer,
            self.optim,
            self.dl,
            self.scheduler
        )

        # self.accelerator.register_for_checkpointing(self.scheduler)
        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = self.hp.save_model_every
        self.save_results_every = self.hp.save_results_every

        self.results_folder = Path(result_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

        hps = {"num_train_steps": self.num_train_steps, "batch_size": self.batch_size}
        self.accelerator.init_trackers(config.hparams.name, config=hps)

    def save(self, path):
        pkg = dict(
            model=self.accelerator.get_state_dict(self.transformer),
            optim=self.optim.state_dict(),
            scheduler=self.scheduler.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path, steps=0):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')

        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.steps = torch.Tensor([steps])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.transformer)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device
        self.transformer.train()

        logs = {}
        self.optim.zero_grad()

        start = datetime.datetime.now()
        steps = int(self.steps.item())
        for _ in range(self.grad_accum_every):
            with self.accelerator.accumulate(self.transformer):
                batch = next(self.dl_iter)
                data_lens, input_datas, pos_ids = self.ds.to_device(batch[0], device)
                logits = self.transformer(input_datas[:, :-1], merge_context=True, position_ids=pos_ids[:, :-1],
                                          training=True)

                # calculate loss
                loss_mask = sequence_mask(data_lens - 1, max_len=None, device=device)

                loss, top3_acc = compute_loss(logits[:, self.max_text_length:, :], input_datas[:,
                                                                                    self.max_text_length + self.max_history_semantic_length + 1:],
                                               loss_mask[:,self.max_text_length + self.max_history_semantic_length:])


                self.accelerator.backward(loss)
                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad()
                self.scheduler.step()

                accum_log(logs, {'loss': loss.item() / self.grad_accum_every})
                accum_log(logs, {'top10_acc': top3_acc.item() / self.grad_accum_every})
        # log
        times = datetime.datetime.now() - start
        self.print(
            f"{steps:d}/{self.num_train_steps}: loss: {logs['loss']}, top10_acc: {logs['top10_acc']}, lr: {self.scheduler.get_last_lr()} {times * (self.num_train_steps - steps)}<{times}s")
        self.accelerator.log({"train_loss": logs['loss'], "top10_acc": logs['top10_acc'],
                              "learning_rate": self.scheduler.get_last_lr()[0].item()},
                             step=steps)

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            model = self.unwrapped_model
            batch = next(self.valid_dl_iter)
            data_lens, input_datas, pos_ids = self.ds.to_device(batch[0], device)
            model.eval()
            with torch.no_grad():
                logits = self.transformer(input_datas[:, :-1], merge_context=True, position_ids=pos_ids[:, :-1],
                                          training=True)
                loss_mask = sequence_mask(data_lens - 1, max_len=None, device=device)

                valid_loss, top10_acc = compute_loss(logits[:, self.max_text_length:, :], input_datas[:,
                                                                                          self.max_text_length + self.max_history_semantic_length + 1:],
                                                     loss_mask[:,self.max_text_length + self.max_history_semantic_length:])
                valid_loss = valid_loss.item()
            self.print(f'{steps}: valid loss {valid_loss}, valid top10_acc {top10_acc.item()}')
            self.accelerator.log({"valid_loss": valid_loss, "valid_top10_acc": top10_acc.item()}, step=steps)
        self.accelerator.wait_for_everyone()
        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'ar.transformer.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self):
        while self.steps < self.num_train_steps:
            logs = self.train_step()

        self.print('training complete')
        self.accelerator.end_training()


if __name__ == '__main__':
    # 训练
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint_step',
                        type=int,
                        default=0,
                        help='Checkpoint step to load')
    args = parser.parse_args()
    config = get_config_from_file(args.config)
    hparams = config.hparams

    model, tokenizer = load_text_to_semantic_model(
        ckpt_path='/data4/leishun/text_to_semantic/pretrained_model/text_2.pt', device='cpu')

    calculate_model_params(model)
    trainer = TextToSemanticTrainer(
        transformer=model,
        config=config
    )

    if args.checkpoint_step > 0:
        trainer.load(os.path.join('checkpoint', config.hparams.name, f'ar.transformer.{args.checkpoint_step}.pt'),
                     steps=args.checkpoint_step)

    trainer.train()
