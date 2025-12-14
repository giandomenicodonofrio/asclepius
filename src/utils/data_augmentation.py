import random
import numpy as np

synthetic_ecg = np.load("./resources/synthetic_ecg.npy")


def neutral(data):
    return data


def scale(data):
    """ 
    Scale signal
    """
    factor = random.uniform(0.5, 2)
    d = data*factor
    return d


def flip(data):
    """ 
    invert signal
    """
    d = data * -1
    return d


def mixup(data1, data2):
    """ 
    sum two signal
    """
    d = data1 + data2
    return d


def add_synthetic_ecg(data):
    """
    add a synthetic ecg to the signal
    """
    k = 0
    d = np.zeros(data.shape)
    for t in range(len(data)):
        if (t % 10 == 0):
            k += 1
        d[t] = data[t] + synthetic_ecg[k % len(synthetic_ecg)]
    return d


def add_sinus(data, amplitude=1, frequency=0.001, theta=0):
    """
    add a sinus function that is similar to ecg to the signal
    """
    d = np.zeros(data.shape)
    for t in range(len(data)):
        d[t] = data[t] + amplitude * np.sin(frequency * 7*t + theta) * np.sin(
            frequency*5*t + theta) * np.cos(frequency*3.25*t + theta)
    return d


def drop(data):
    """ 
    drop some random points to zero
    """
    d = np.copy(data)
    points = np.arange(data.shape[0])
    np.random.shuffle(points)
    num_pointes = random.choice(np.arange(data.shape[0]//3))
    for t in points[:num_pointes]:
        d[t, :] = 0
    return d


def time_masking(data):
    """ 
    drop a random time window (maximum 20%) of signal to the zero
    """
    cutoff_len = random.choice(np.arange(data.shape[0]//5))
    points = np.arange(data.shape[0])
    np.random.shuffle(points)
    start_point = points[0]
    del points

    end_point = start_point + cutoff_len
    if end_point >= data.shape[0]:
        end_point = -1

    d = np.copy(data)
    d[start_point: end_point, :] = 0
    return d


def shift(data):
    """
    shif signal in random direction and random quantity (maximum 20%)
    """
    shift_len = random.choice(np.arange(data.shape[0]//5))
    direction = random.choice([-1, 1])
    d = np.copy(data)
    d = np.roll(d, direction * shift_len, axis=0)
    return d


probs = [0.1, 0.3, 0.3, 0.2, 0.1]

# Initialize transformations
transformations = [
    neutral,
    shift,
    time_masking,
    drop,
    add_synthetic_ecg
]

class Augmenter:
    def __init__(self, transformations, probs):
        self.transformations = transformations
        self.probs = probs

    def __call__(self, batch):
        random_tranformations = random.choices(
            self.transformations, weights=self.probs, k=batch.shape[0])
        return np.array([transformation(sample) for sample, transformation in zip(batch, random_tranformations)])

def get_default_augmenter():
    return Augmenter(transformations, probs)