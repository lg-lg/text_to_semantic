import torchaudio
from hubert_kmeans import HubertWithKmeans
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)

from pathlib import Path

# 先获取语音的数据
folder_list = ['/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/dev-clean','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/dev-other','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/test-clean','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/test-other','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/train-clean-100','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/train-clean-360','/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS/train-other-500']
suffix = '.wav'
speech_list = []
for folder in folder_list:
    folder = Path(folder)
    path = [*folder.rglob(f'*{suffix}')]
    speech_list.extend(path)

print(len(speech_list))

from tqdm import tqdm
import os
import sys

no_speech_file = []
for data in tqdm(speech_list[350000:]):
    wav_path = data
    sys.stdout = open(os.devnull, 'w')
    result = inference_pipeline(audio_in=str(wav_path))
    sys.stdout = sys.__stdout__
    if 'text' not in result:
        no_speech_file.append(wav_path)
        continue
    result = result['text']
    result = np.array(result)
    save_path = str(wav_path).replace('/data4/nas40T/share/shun.lei/workspace/data/raw_data/LibriTTS','/data4/leishun/processed_data/speech_vad_result').replace('.wav','.npy')
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    np.save(save_path,result)

print(no_speech_file)
    # start = result[0][0]
    # end = result[-1][-1]
    # duration = (end-start)
    # cnt = 0
    # for time in result:
    #     cnt += time[1]-time[0]
    # rate = cnt/duration
    # speech_rate.append(rate)
