import numpy as np
from scipy.signal import butter, savgol_filter, lfilter, filtfilt, sosfiltfilt
from src.utils.visualization import ECGplot

# FS = 100  # corresponds to 60 beats per min (normal for human), assumed.
# LOWCUT = 0.01  # 9.9 beats per min
# HIGHCUT = 17  # 900 beats per m

SAMPLING_FREQUENCY = 500.0
LOWCUT = 1.0/(SAMPLING_FREQUENCY/2)
HIGHCUT = 47.0/(SAMPLING_FREQUENCY/2)

def set2zero_lead(data, lead_ids):
    """
    Put to the zero spefic leads (lead_ids)
    """
    zeroIndices = np.asarray(list(set(range(12)) - set([lead_ids])))
    data[:,zeroIndices] = 0

def ldasg_filter(data, window_length = 5, polyorder=2):
    return savgol_filter(data, window_length, polyorder)

def butter_bandpass(lowcut, highcut, order=3):
    return butter(order, [lowcut, highcut], 'bandpass')

def butterworth_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, order=3):
    """
    Denoising using band pass filter
    """
    b, a = butter_bandpass(lowcut, highcut, order=order)
    return filtfilt(b, a, data.T).T

def zscore_normalization(data, mean=0, dev=0.199):
    """
    Perform normalization
    """
    return (data - mean) / dev

def preprocessing_func(batch, lowcut=LOWCUT, highcut=HIGHCUT, mean=0, dev=0.199):
    for i, d in enumerate(batch):
        d = zscore_normalization(d, mean, dev)
        d = butterworth_filter(d, lowcut, highcut)
        batch[i] = d
    return batch

class Preprocesser:
    def __init__(self, preprocessing_func, preprocessing_args):
        self.func = preprocessing_func
        self.args = preprocessing_args
    
    def __call__(self, batch):
        return self.func(batch, **self.args)

def get_default_preprocesser():
    return Preprocesser(preprocessing_func, {})
