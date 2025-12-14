from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import math
import pickle
import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
from pathlib import Path
from reference.architecture import *
from src.utils.data import generator
from hyperband.hyperband_base_model import CustomHistory
from src.utils.utility import path_creator
import os

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


class EnsembleModel:

    def get_history_path(self, combination):
        return self._get_sub_model_saved_path(combination).joinpath('history')
        

    def _get_model_saved_path(self, base_path=None):
        return Path(self.model_path if base_path is None else base_path, f'fold{self.id}/')

    def _get_sub_model_saved_path(self, lead_combination, base_path=None):
        return self._get_model_saved_path(base_path).joinpath( f'lead_combination_{"-".join(map(str, lead_combination))}')


    def __init__(self, id, preprocesser = None, augmenter=None, lead_configuration=[(12,)], model_path="ensemble_model", magic_vector_path="resources/magic_vector.npy"):
        self.model_path = model_path
        self.magic_vector_path = magic_vector_path
        self.magic_vector = np.load(magic_vector_path)
        self.lead_configuration = lead_configuration
        self.models = []
        self.id = id
        self.preprocesser = preprocesser
        self.augmenter = augmenter

        path_creator(model_path)

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

    def fit(self, X_pointer, y_pointer, train_idx, val_idx, epochs, batch_size=64):
        train_sample_count = len(train_idx)
        val_sample_count = len(val_idx)
        custom_h = CustomHistory()
        for combination in self.lead_configuration:
            if os.path.exists(self.get_history_path(combination)):
                print(f"Combination - {'-'.join(map(str, combination))} already trained")
                continue
            print(
                f'Train combination - {"-".join(map(str, combination))}')
            h = None
            model_checkpoint = ModelCheckpoint(
                filepath=f'{self._get_sub_model_saved_path(combination)}/cp.ckpt',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_best_only=True)
            early_stopping_callback = EarlyStopping(
                monitor='val_loss', patience=15, verbose=1, mode='min')


            model = get_compiled_architecture()

            train_generator = generator(batch_size, train_idx, list(combination), True, self.preprocesser, self.augmenter, X_pointer, y_pointer)
            val_generator = generator(batch_size, val_idx, list(combination), False, self.preprocesser, self.augmenter, X_pointer, y_pointer)

            h = model.fit(train_generator,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=val_generator,
                          steps_per_epoch=math.ceil(
                              train_sample_count / batch_size),
                          validation_steps=math.ceil(
                              val_sample_count / batch_size),
                          callbacks=[early_stopping_callback,
                                     model_checkpoint, LearningRateScheduler(scheduler)],
                          max_queue_size=16,
                          verbose=2)

            symbolic_weights = getattr(model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            optimizer_path = f'{self._get_sub_model_saved_path(combination)}/optimizer.pkl'
            path_creator(optimizer_path)
            with open(optimizer_path, 'wb') as f:
                pickle.dump(weight_values, f)

            max_val_acc_idx = np.argmax(np.array(h.history["val_accuracy"]))
            custom_h.history["accuracy"].append(
                h.history["accuracy"][max_val_acc_idx])
            custom_h.history["loss"].append(h.history["loss"][max_val_acc_idx])
            custom_h.history["val_accuracy"].append(
                h.history["val_accuracy"][max_val_acc_idx])
            custom_h.history["val_loss"].append(
                h.history["val_loss"][max_val_acc_idx])
            del model
            gc.collect()
            K.clear_session()
            tf.compat.v1.reset_default_graph()


            history_path = f'{self._get_sub_model_saved_path(combination)}/history'
            path_creator(history_path)
            with open(history_path, "wb") as file_pi:
                pickle.dump(h.history, file_pi)
        if len(custom_h.history) == 0:
            return None
        return custom_h

    def load_model(self, models_path=None):
        for combination in self.lead_configuration:
            model = get_compiled_architecture()
            base_path = self._get_sub_model_saved_path(combination, base_path=models_path)
            checkpoint_path = base_path.joinpath("cp.ckpt")
            model.load_weights(checkpoint_path)
            self.models.append(model)
        return True

    def predict(self, X_pointer, test_idx, batch_size=64):
        if len(self.models) == 0:
            if not self.load_model():
                return None
        sample_count = len(test_idx)
        y_preds = []
        for i, combination in enumerate(self.lead_configuration):
            print(f'Predict combination - {"-".join(map(str, combination))}')

            test_generator = generator(batch_size, test_idx, list(combination), False, self.preprocesser, self.augmenter, X_pointer)

            local_y_preds = self.models[i].predict(
                test_generator, batch_size=batch_size, steps=math.ceil(sample_count / batch_size))
            y_preds.append(local_y_preds)
            del local_y_preds

        y_preds = np.array(y_preds)
        y_preds = np.mean(y_preds, axis=0)
        y_preds = self.preds2onehot(y_preds)
        return y_preds
