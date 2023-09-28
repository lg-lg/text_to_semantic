import torchaudio
import multiprocessing
import json
from tqdm import tqdm
import os
from pathlib import Path
from encodec.utils import convert_audio


def save_wavs(music):
    wav_path = music[0]['wav']
    wav_path = Path(wav_path)
    if not wav_path.exists():
        print(wav_path)
        return
    data, sr = torchaudio.load(wav_path)
    for m in music:
        start = m['start']
        end = m['end']
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        if start_frame >= end_frame:
            continue
        seg = data[:, start_frame:end_frame]
        if seg.shape[-1] == 0:
            print(m)
            continue
        torchaudio.save(os.path.join('/data4/leishun/processed_data/cut_vocal', m['utt'] + '.wav'), seg, sr)
        seg16k = convert_audio(seg, sr, 16000, 1)
        torchaudio.save(os.path.join('/data4/leishun/processed_data/cut_vocal_16k', m['utt'] + '.wav'), seg16k, 16000)
        seg24k = convert_audio(seg, sr, 24000, 1)
        torchaudio.save(os.path.join('/data4/leishun/processed_data/cut_vocal_24k', m['utt'] + '.wav'), seg24k, 24000)


data_list = []
with open('/data4/leishun/text_to_semantic/datasets/new_segment.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)
# 打印读取到的数据
for item in data_list[:10]:
    print(item)

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

with multiprocessing.Pool(processes=32) as pool:
    for _ in tqdm(pool.imap_unordered(save_wavs, music_list), total=len(music_list)):
        pass

# for k in tqdm(music_data.keys()):
#     music = music_data[k]
#     wav_path = music[0]['wav']
#     data, sr = torchaudio.load(wav_path)
#     for m in music:
#         start = m['start']
#         end = m['end']
#         start_frame = int(start * sr)
#         end_frame = int(end * sr)
#         seg = data[:, start_frame:end_frame]
#         torchaudio.save(os.path.join('/data4/leishun/processed_data/raw_cut_vocal',m['utt']+'.wav'),seg,sr)
