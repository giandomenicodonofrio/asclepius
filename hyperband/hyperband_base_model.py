from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict

class CustomHistory():
    def __init__(self):
        self.history = defaultdict(list)

class HyperBandBaseModel(ABC):
    
    def _one_hot(self, class_ind):
        ris = np.zeros(9)
        ris[class_ind] = 1
        return ris

    def preds2onehot(self, y_preds):
        y_preds = np.argmax(y_preds, axis=1)
        y = []
        for y_p in y_preds:
            y.append(self._one_hot(y_p))
        del y_preds
        return np.array(y)

    @abstractmethod
    def __init__(self, conf, id, i):
        pass

    @abstractmethod
    def fit(self, X_pointer, y_pointer, train_idx, val_idx, epochs, batch_size):
        pass

    @abstractmethod
    def predict(self, X_pointer, test_idx, batch_size):
        pass

