import torchaudio
from hubert_kmeans import HubertWithKmeans
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import json

# 存放解析后的 JSON 数据
data_list = []
with open('/data4/leishun/text_to_semantic/datasets/new_segment.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)
# 打印读取到的数据
for item in data_list[:10]:
    print(item)

cnt = 0
for data in data_list:
    cnt += float(data['dur'])
print(cnt)
# inference_pipeline = pipeline(
#     task=Tasks.voice_activity_detection,
#     model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
#     model_revision=None,
# )
# segments_result = inference_pipeline(audio_in='/data4/leishun/processed_data/vad_vocal_16k/NGJgWTt8TOo-30.wav')
# print(segments_result)


# data ,sr = torchaudio.load('/data4/leishun/text_to_semantic/vocals.opus')
# data = data[:,:48000*20]

# def load_model(device):
#     wav2vec = HubertWithKmeans(
#         hubert_path='/data4/leishun/text_to_semantic/hubert_checkpoint/hubert_base_ls960.pt',
#         kmeans_path='/data4/leishun/text_to_semantic/hubert_checkpoint/quantifier_hubert_base_ls960_14.pth',
#         target_sample_hz=16000,
#         seq_len_multiple_of=320,
#         device='cuda'
#     )
#     return wav2vec.to(device)

# model = load_model('cuda')

# semantic_token_ids = model(data, sr=sr)
# print(semantic_token_ids)