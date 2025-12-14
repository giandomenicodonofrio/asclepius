import h5py 
import numpy as np 
import csv
from tqdm import tqdm
import scipy.io as sio
from threading import Thread
import os
from collections import defaultdict
import json
from pathlib import Path

single_class = defaultdict(list)
double_class = defaultdict(list)
triple_class = defaultdict(list)

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

class_map = [single_class, double_class, triple_class]

DATASET_PATH = ""
LABELS_PATH = ""
DIVIDED_DATASET_PATH = ""


def save(file_name, datasets, dataset_names, append=False):
    with h5py.File(file_name, "a" if append else "w") as f:
        for dataset, name in zip(datasets, dataset_names): 
            f.create_dataset(name, data=dataset)

def one_hot(labels):
    ris = np.zeros(9)
    ris[labels] = 1
    return ris

def job(data, label, output_path, dataset_path = DATASET_PATH):
    """ 
    Load data -> resize to 72000, label to onehot -> append to dataset -> save on file
    """
    dataset = ([], [])
    for d in data:
        path = os.path.join(dataset_path, d + ".mat")
        ecg = np.zeros((72000,12), dtype=np.float32)
        sample = sio.loadmat(path)['ECG'][0][0][2]
        ecg[-sample.shape[1]:,:] = sample[:, -72000:].T
        dataset[0].append(ecg)
        dataset[1].append(one_hot(label))
    os.chdir(output_path)
    save(f"dataset_divived_label_{label}", datasets=list(dataset), dataset_names=["X", "y"])

def merge(output_path, divided_datasets_path = DIVIDED_DATASET_PATH):
    """
    Merge divided dataset in X, make dataset referenced to class in X/i and y/i
    """
    with h5py.File(output_path,"w") as f:
        for i in tqdm(range(1, 10)):
            with h5py.File(os.path.join(divided_datasets_path, f'dataset_divived_label_{i}'), 'r') as f2:
                tmp = f2['X'][...]
                f.create_dataset(f'X/{i}', data=tmp)
                tmp = f2['y'][...]
                f.create_dataset(f'y/{i}', data=tmp)






def job(job_id, data, output_dir, dataset_path):
    """
    load ecg
    resize ecg to 72000
    append ecg to dataset with label
    append id to a map of classes

    save dataset
    save map
    """
    dataset = ([], [])
    dataset_info = []
    sample_label_map = defaultdict(list)
    for d in tqdm(data):
        index, sample_id, classes, label = d
        sample_path = os.path.join(dataset_path, sample_id + ".mat")
        ecg = np.zeros((72000,12), dtype=np.float32)
        sample = sio.loadmat(sample_path)['ECG'][0][0][2]
        ecg[-sample.shape[1]:,:] = sample[:, -72000:].T

        dataset[0].append(ecg)
        dataset[1].append(label)
        sample_label_map[str(label)].append(index)
        means = sample.mean(axis=1)
        devs = sample.std(axis=1)
        dataset_info.append({
            "id": sample_id,
            "index": index,
            "classes": classes,
            "label": label,
            "length": len(sample),
            "means": means.tolist(),
            "devs": devs.tolist(),
            "mean_dev": np.array(devs).mean()
        })
        
    save(os.path.join(output_dir, f"dataset_divided_label_job_{job_id}"), datasets=list(dataset), dataset_names=["X", "y"])
    p = Path(os.path.join(output_dir,f'sample_label_map_job_{job_id}.json'))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir,f'sample_label_map_job_{job_id}.json'), 'w') as fp:
        json.dump(sample_label_map, fp)
    p = Path(os.path.join(output_dir,f'dataset_info_job_{job_id}.json'))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir,f'dataset_info_job_{job_id}.json'), 'w') as fp:
        x = json.dumps(dataset_info)
        json.dump(x, fp)

def extract_label_from_classes(classes):
    classes_sorted = sorted(classes, key=lambda i: class_sample[i])[0]
    return classes_sorted


def get_id_reference_map(reference_path):
    reference_map=[]
    with open(reference_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for i, rows in enumerate(reader):
            id = rows[0]
            classes = list(filter(lambda lab : lab !="", rows[1:]))
            label = extract_label_from_classes(classes)
            reference_map.append((i, id, classes, label))
    return reference_map

""" ref_map = get_id_reference_map("./reference_dataset.csv")
print(ref_map) """


def dataset_creation(dataset_info, dataset_path, output_path):
    classes_ids = defaultdict(list)
    X = []
    y = []
    for d in tqdm(dataset_info):
        index, sample_id, classes, label = d
        sample_path = os.path.join(dataset_path, sample_id + ".mat")
        ecg = np.zeros((72000,12), dtype=np.float32)
        sample = sio.loadmat(sample_path)['ECG'][0][0][2]
        ecg[-sample.shape[1]:,:] = sample[:, -72000:].T
        X.append(ecg)
        y.append(one_hot(int(label)-1))
        classes_ids[label].append(index)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(f'{output_path}/dataset', "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        for key in classes_ids.keys():
            f.create_dataset(f"x/{key}", data=classes_ids[key])


def main():
    """
    Load labels
    Divide jobs
    Merge results
    """
    ref_map = get_id_reference_map("./resources/reference_dataset.csv")
    dataset_creation(ref_map, "resources/dataset", "out_dataset")

main()