import torchaudio
import json
from tqdm import tqdm
import os
from pathlib import Path
from encodec.utils import convert_audio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from encodec import EncodecModel
from transformers import BertTokenizer
import torch
import subprocess as sp

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
    acoustic_model = load_encodec(device)

    ds = Data(meta_file)
    dl = DataLoader(ds, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x[0])

    (acoustic_model, dl) = accelerate.prepare(acoustic_model, dl)

    for batch in tqdm(dl):
        with torch.no_grad():
            wav_path = batch[0]['wav'].replace('separated_opus/htdemucs/', 'audio/').replace('/vocals.opus', '.opus')
            wav_path = Path(wav_path)
            if not wav_path.exists():
                print(wav_path)
                continue
            data, sr = torchaudio.load(wav_path)
            if data.shape[0] != 1 and data.shape[0] != 2:
                print(wav_path)
                # 先用ffmpeg转
                new_file = wav_path.with_suffix('.wav')
                new_file = Path(str(new_file).replace('/data4/music-corpus/yt-best/audio', '/home/shun.lei/audio'))
                new_file.parent.mkdir(exist_ok=True, parents=True)
                command = ['ffmpeg', '-i', str(wav_path), '-acodec', 'pcm_s16le', '-ac' ,'1' ,'-ar', '48000', str(new_file), '-y']
                    # ,'-loglevel', 'quiet']
                sp.run(command)

                data, sr = torchaudio.load(new_file)

            for m in batch:
                start = float(m['start'])
                end = float(m['end'])
                start_frame = int(start * sr)
                end_frame = int(end * sr)

                seg = data[:, start_frame:end_frame]
                if seg.shape[-1] < sr * 0.1:
                    continue

                # 提取acoustic

                seg_24k = convert_audio(seg, sr, 24000, 1).to(device).unsqueeze(0)

                acoustic_token_ids = acoustic_model.module.encode(seg_24k)
                acoustic_token_ids = acoustic_token_ids[0][0][0].cpu()  # (8, t)

                vocal_acoustic_token_ids = torch.load(
                    os.path.join('/data4/leishun/processed_data/vocal_acoustic', m['utt'] + '.npy'))
                acosutic_len = vocal_acoustic_token_ids.shape[-1]

                if acosutic_len > acoustic_token_ids.shape[-1]:
                    acoustic_token_ids = torch.cat([acoustic_token_ids, acoustic_token_ids[:, -1].unsqueeze(1).repeat(1,
                                                                                                                      acosutic_len -
                                                                                                                      acoustic_token_ids.shape[
                                                                                                                          -1])],
                                                   dim=-1)
                else:
                    acoustic_token_ids = acoustic_token_ids[:, :acosutic_len]
                assert acoustic_token_ids.shape[-1] % 3 == 0 and acoustic_token_ids.shape[-1] == acosutic_len

                # 存储
                outpath = os.path.join('/data4/leishun/processed_data/song_acoustic', m['utt'] + '.npy')
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                torch.save(acoustic_token_ids, outpath)
