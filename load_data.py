import torchaudio
from hubert_kmeans import HubertWithKmeans
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys

folder_list = ['/data4/leishun/processed_data/speech_vad_result']
suffix = '.npy'
file_list = []
for folder in folder_list:
    folder = Path(folder)
    path = [*folder.rglob(f'*{suffix}')]
    file_list.extend(path)
vad_results = []
for file in tqdm(file_list):
    result = np.load(file)
    vad_results.append(result)
import numpy as np
from multiprocessing import Pool

def load_npy(file_path):
    return np.load(file_path)


with Pool(processes=32) as pool:  # 创建一个包含4个进程的进程池
    data_list = pool.map(load_npy, file_list)
