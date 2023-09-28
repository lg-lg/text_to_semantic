from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
import numpy as np


def pad(input_ele, mel_max_length=None, pad_value=0.0):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", pad_value
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", pad_value
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class TextSemanticDataset(Dataset):
    def __init__(self,
                 meta_file,
                 max_text_length=256,
                 max_history_semantic_length=256,
                 max_semantic_length=768,
                 semantic_pad=10000,
                 text_pad=129595,
                 semantic_infer=129599,
                 text_offset=10048,
                 ):
        super().__init__()
        data_list = []
        with open(meta_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data['utt'])

        # 加载所有数据进内存
        datas = []
        for name in tqdm(data_list):
            data = []
            semantic_tokens = torch.load(os.path.join('/data4/leishun/processed_data/vocal_semantic', name + '.npy'))
            history_semantic_tokens = torch.load(
                os.path.join('/data4/leishun/processed_data/vocal_history_semantic', name + '.npy'))
            text_tokens = torch.load(os.path.join('/data4/leishun/processed_data/vocal_text', name + '.npy'))

            if history_semantic_tokens.shape[0] > max_history_semantic_length:
                history_semantic_tokens = history_semantic_tokens[-max_history_semantic_length:]
            else:
                history_semantic_tokens = torch.cat([history_semantic_tokens, torch.ones(
                    max_history_semantic_length - history_semantic_tokens.shape[0]).long() * semantic_pad])

            if semantic_tokens.shape[0] >= max_semantic_length:
                semantic_tokens = semantic_tokens[:max_semantic_length]
            else:
                semantic_tokens = torch.cat([semantic_tokens, torch.ones(1).long() * semantic_pad])

            data.append(text_tokens)
            data.append(history_semantic_tokens)
            data.append(semantic_tokens)
            datas.append(data)

        self.datas = pd.DataFrame(datas, columns=['text', 'history_semantic', 'semantic'])

        self.max_text_length = max_text_length
        self.max_history_semantic_length = max_history_semantic_length
        self.max_semantic_length = max_semantic_length
        self.semantic_pad = semantic_pad
        self.text_pad = text_pad
        self.semantic_infer = semantic_infer
        self.text_offset = text_offset

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas.loc[item]
        text = data['text']
        history_semantic = data['history_semantic']
        semantic = data['semantic']
        sample = {
            'text': text,
            'history_semantic': history_semantic,
            'semantic': semantic
        }

        return sample

    def _prepare_batch(self, data, idx):
        text_datas = [data[i]['text'] for i in idx]
        history_semantic_datas = [data[i]['history_semantic'] for i in idx]
        semantic_datas = [data[i]['semantic'] for i in idx]

        input_datas = []
        pos_ids = []

        for text_data, history_semantic_data, semantic_data in zip(text_datas, history_semantic_datas, semantic_datas):
            text_len = text_data.shape[0]
            history_semantic_len = history_semantic_data.shape[0]
            semantic_len = semantic_data.shape[0]
            text_data = text_data + self.text_offset
            if text_data.shape[0] > self.max_text_length:
                text_data = text_data[:self.max_text_length:]
            else:
                text_data = torch.cat(
                    [text_data, torch.ones(self.max_text_length - text_len).long() * self.text_pad])
            assert text_data.shape[0] == self.max_text_length
            assert history_semantic_len == self.max_history_semantic_length

            semantic_infer_token = (torch.ones(1).long() * self.semantic_infer)

            input_data = torch.cat([text_data, history_semantic_data, semantic_infer_token, semantic_data],
                                   dim=0)  # (t, )
            pos_id = torch.arange(input_data.shape[0] - self.max_history_semantic_length).long()  # (t, )

            input_datas.append(input_data)
            pos_ids.append(pos_id)

        data_lens = np.array([input_data.shape[0] for input_data in input_datas])
        input_datas = pad(input_datas, pad_value=self.semantic_pad)
        pos_ids = pad(pos_ids, pad_value=0)

        a_batch = (data_lens, input_datas, pos_ids)
        return a_batch

    def to_device(self, data, device):
        (data_lens, input_datas, pos_ids) = data
        data_lens = torch.from_numpy(data_lens).to(device)
        input_datas = input_datas.to(device)
        pos_ids = pos_ids.to(device)

        a_device_batch = (data_lens, input_datas, pos_ids)
        return a_device_batch

    def collate_fn(self, data):
        data_size = len(data)

        idx_arr = np.arange(data_size)


        tail = idx_arr[len(idx_arr) - (len(idx_arr) % data_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % data_size)]
        idx_arr = idx_arr.reshape((-1, data_size)).tolist()

        output = list()
        for idx in idx_arr:
            output.append(self._prepare_batch(data, idx))

        return output
