import torchaudio
import multiprocessing
import json
from tqdm import tqdm
import os
from pathlib import Path
from encodec.utils import convert_audio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from hubert_kmeans import HubertWithKmeans
from encodec import EncodecModel
from transformers import BertTokenizer
import torch


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def load_hubert(device):
    wav2vec = HubertWithKmeans(
        hubert_path='/data4/leishun/text_to_semantic/hubert_checkpoint/hubert_base_ls960.pt',
        kmeans_path='/data4/leishun/text_to_semantic/hubert_checkpoint/quantifier_hubert_base_ls960_14.pth',
        target_sample_hz=16000,
        seq_len_multiple_of=320,
        device='cuda'
    )
    return wav2vec.to(device)


def load_encodec(device):
    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    return model


class Data(Dataset):
    def __init__(self, meta_file):
        data_list = []
        with open(meta_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)
        # 按music名字分类
        music_data = {}
        for data in data_list:
            name = '-'.join(data['utt'].split('-')[:-1])
            if name not in music_data:
                music_data[name] = []
            music_data[name].append(data)

        music_list = []
        for k in tqdm(music_data.keys()):
            music = music_data[k]
            music_list.append(music)
        self.music_list = music_list

    def __len__(self):
        return len(self.music_list)

    def __getitem__(self, item):
        return self.music_list[item]


if __name__ == '__main__':
    accelerate = Accelerator()
    meta_file = '/data4/leishun/text_to_semantic/datasets/new_segment.json'
    device = accelerate.device
    num_workers = 8
    semantic_model = load_hubert(device)
    acoustic_model = load_encodec(device)
    tokenizer = BertTokenizer.from_pretrained(
        '/home/shun.lei/.cache/huggingface/hub/models--bert-base-multilingual-cased')

    ds = Data(meta_file)
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0])

    (semantic_model, acoustic_model, dl) = accelerate.prepare(semantic_model, acoustic_model, dl)

    for batch in tqdm(dl):
        with torch.no_grad():
            wav_path = batch[0]['wav']
            wav_path = Path(wav_path)
            if not wav_path.exists():
                print(wav_path)
                continue
            data, sr = torchaudio.load(wav_path)
            for m in batch:
                start = float(m['start'])
                end = float(m['end'])
                history_start = max(0, start - 5.2)
                start_frame = int(start * sr)
                end_frame = int(end * sr)
                history_start_frame = int(history_start * sr)

                seg = data[:, start_frame:end_frame]
                if seg.shape[-1] < sr * 0.1:
                    continue

                # 提取semantic
                semantic_token_ids = semantic_model(seg.to(device), sr=sr).cpu()  # (t,)
                # 提取历史的semantic token
                if start - history_start <= 0.1:
                    history_token_ids = torch.tensor([10000])
                else:
                    history_seg = data[:, history_start_frame:start_frame].to(device)
                    history_token_ids = semantic_model(history_seg, sr=sr).cpu()
                # 提取acoustic
                seg_24k = convert_audio(seg, sr, 24000, 1).to(device).unsqueeze(0)
                acoustic_token_ids = acoustic_model.module.encode(seg_24k)
                # acoustic_token_ids = acoustic_model.encode(seg_24k)
                acoustic_token_ids = acoustic_token_ids[0][0][0].cpu()  # (8, t)
                # 提取文本id
                text = m['text']
                text_token_ids = _tokenize(tokenizer, text)
                text_token_ids = torch.tensor(text_token_ids)
                # 长度对齐2：3
                # 规整semantic到2的整数倍（pad1或者2）
                if semantic_token_ids.shape[0] % 2 == 1:
                    semantic_token_ids = F.pad(semantic_token_ids, (0, 1), value=semantic_token_ids[-1])
                else:
                    semantic_token_ids = F.pad(semantic_token_ids, (0, 2), value=semantic_token_ids[-1])
                # 规整acoustic到3的整数倍
                acosutic_len = int(semantic_token_ids.shape[0] / 2 * 3)
                if acosutic_len > acoustic_token_ids.shape[-1]:
                    acoustic_token_ids = torch.cat([acoustic_token_ids, acoustic_token_ids[:, -1].unsqueeze(1).repeat(1,
                                                                                                                      acosutic_len -
                                                                                                                      acoustic_token_ids.shape[
                                                                                                                          -1])],
                                                   dim=-1)
                else:
                    acoustic_token_ids = acoustic_token_ids[:, :acosutic_len]
                assert acoustic_token_ids.shape[-1] % 3 == 0 and semantic_token_ids.shape[0] % 2 == 0 and \
                       acoustic_token_ids.shape[-1] / 3 == semantic_token_ids.shape[0] / 2

                # 存储
                # 存semantic
                outpath = os.path.join('/data4/leishun/processed_data/vocal_semantic', m['utt'] + '.npy')
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(semantic_token_ids, outpath)

                # 存history semantic
                outpath = os.path.join('/data4/leishun/processed_data/vocal_history_semantic', m['utt'] + '.npy')
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(history_token_ids, outpath)

                # 存文本id
                outpath = os.path.join('/data4/leishun/processed_data/vocal_text', m['utt'] + '.npy')
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(text_token_ids, outpath)

                # 存acoustic
                outpath = os.path.join('/data4/leishun/processed_data/vocal_acoustic', m['utt'] + '.npy')
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(acoustic_token_ids, outpath)
