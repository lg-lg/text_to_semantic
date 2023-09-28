import torchaudio
from hubert_kmeans import HubertWithKmeans
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)

# 先获取语音的数据
data_list = []
with open('/data4/leishun/text_to_semantic/datasets/data.segment.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)

# 打印读取到的数据
for item in data_list[:10]:
    print(item)

no_speech_file = []
for data in tqdm(data_list[800000:]):
    wav_path = os.path.join('/data4/leishun/processed_data/vad_vocal_16k', data['utt'] + '.wav')
    if not Path(wav_path).exists():
        no_speech_file.append(wav_path)
        continue
    save_path = str(wav_path).replace('/vad_vocal_16k',
                                      '/vocal_vad_result').replace('.wav', '.npy')
    save_path = Path(save_path)
    if save_path.exists():
        continue
    if data['end'] - data['start'] < 0.1 or data['dur'] < 0.1:
        no_speech_file.append(wav_path)
        continue
    try:
        result = inference_pipeline(audio_in=str(wav_path))
    except:
        no_speech_file.append(wav_path)
        continue
    if 'text' not in result:
        no_speech_file.append(wav_path)
        continue
    result = result['text']
    result = np.array(result)

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    np.save(save_path, result)

print(no_speech_file)
# start = result[0][0]
# end = result[-1][-1]
# duration = (end-start)
# cnt = 0
# for time in result:
#     cnt += time[1]-time[0]
# rate = cnt/duration
# speech_rate.append(rate)
