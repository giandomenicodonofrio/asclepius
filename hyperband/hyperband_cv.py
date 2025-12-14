from keras.callbacks import ModelCheckpoint, EarlyStopping

from collections import defaultdict
import math

from src.utils.data import dataset_division, generator

import itertools
import numpy as np

class HyperBandCV():
    def __init__(self, hypermodel_class,
                 param_map,
                 objective='val_accuracy',
                 preprocessing = None,
                 preprocessing_args_keys=[],
                 max_epochs=10,
                 factor=3,
                 k=10,
                 project_name='hyperbandcv_proj'):
        self.hypermodel_class = hypermodel_class
        self.preprocessing = preprocessing
        self.preprocessing_args_keys = preprocessing_args_keys
        self.objective = objective
        self.max_epochs = max_epochs
        self.param_map = param_map
        self.factor = factor
        self.k = k
        self.project_name = project_name

    def _combination(self, param_map):
        keys, values = zip(*param_map.items())
        permutations_dicts = [dict(zip(keys, v))
                              for v in itertools.product(*values)]
        return permutations_dicts

    def fit(self,X_pointer, y_pointer, data_idx_by_class, batch_size=64):
        prev_epochs = 0
        epochs = 1
        dataset_splitted = dataset_division(
            data_idx_by_class, self.k, flat=True)

        configurations = self._combination(self.param_map)
        conf_enabled = [i for i in range(0, len(configurations))]

        while len(conf_enabled) > 1:
            if epochs > self.max_epochs:
                break
            print(f'\n\n------ Configuration_count = {len(conf_enabled)} ------')
            scores = []
            for conf_idx in conf_enabled:
                conf = configurations[conf_idx]
                conf["preprocessing_args"] = {k:v for k,v in conf.items() if k in self.preprocessing_args_keys}
                conf_score = []
                print(f'\n----- conf = {conf} -----\n')
                for fold_idx, (training_idx, val_idx) in enumerate(dataset_splitted):
                    print(f'--- FOLD {fold_idx} ---\n')
                    training_idx = training_idx[:100]
                    val_idx = val_idx[:5]
                    m = self.hypermodel_class(conf, conf_idx, fold_idx, self.preprocessing)  # forse la i non serve
                    h = m.fit(X_pointer, y_pointer, training_idx, val_idx, epochs, prev_epochs, batch_size)
                    max_val_acc = max(h.history["val_accuracy"])
                    print(f'fold {fold_idx} val_acc={max_val_acc}')
                    conf_score.append(max_val_acc)
                conf_score = np.array(conf_score)
                scores.append((conf_idx, conf_score.mean()))

            sorted_final_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            print(sorted_final_scores)

            new_conf_count = len(conf_enabled) // self.factor
            new_conf_count = 1 if new_conf_count == 0 else new_conf_count

            print(f'new conf count is {new_conf_count}')

            conf_enabled = [x[0] for x in sorted_final_scores[:new_conf_count]]
            print(conf_enabled)
            prev_epochs = epochs
            epochs *= 2

        conf = configurations[conf_enabled[0]]
        # final_model = self.hypermodel_class(conf, -1, -1)
        # h = final_model.fit(X_pointer, y_pointer, epochs, batch_size)

        return conf # , h, final_model, 
