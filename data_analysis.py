import os
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import csv
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

dataset_path = "resources/dataset"
reference_path = "resources/reference_dataset.csv"
dataset_info_path = "resources/dataset_info.pickle"

class_sample = {
    "1": 918,
    "2": 1098,
    "3": 704,
    "4": 207,
    "5": 1695,
    "6": 556,
    "7": 672,
    "8": 825,
    "9": 202,
}

def extract_label_from_classes(classes):
    classes_sorted = sorted(classes, key=lambda i: class_sample[i])[0]
    return classes_sorted

def scan():
    dataset_info = []
    with open(reference_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for i, rows in tqdm(enumerate(reader), total=6877):
            sample_id = rows[0]
            classes = list(filter(lambda lab : lab !="", rows[1:]))
            label = extract_label_from_classes(classes)
            sample_path = os.path.join(dataset_path, sample_id + ".mat")
            sample = sio.loadmat(sample_path)['ECG'][0][0][2].T
            length = sample.shape[0]
            means = sample.mean(axis=0)
            devs = sample.std(axis=0)
            dataset_info.append({
                "id": sample_id,
                "index": i,
                "classes": classes,
                "label": label,
                "length": length,
                "means": means.tolist(),
                "devs": devs.tolist()
            })
    p = Path(dataset_info_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_info_path, 'wb') as f:
        pickle.dump(dataset_info, f, pickle.HIGHEST_PROTOCOL)
    return dataset_info

def load_dataset_info():
    with open(dataset_info_path, 'rb') as f:
        return pickle.load(f)

dataset_info = load_dataset_info() if os.path.exists(dataset_info_path) else scan()
dataset_means = np.array([item["devs"] for item in dataset_info])
fig, ax = plt.subplots(6, 2)
fig.suptitle(f"means Histogram")
for i in range(12):
    j = 0
    if i >5:
        j = 1
    ax[i % 6, j].hist([d[0] for d in dataset_means])

plt.show()
