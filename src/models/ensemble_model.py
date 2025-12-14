from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import math
import pickle
import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
from pathlib import Path
from reference.architecture import *
from src.utils.data import generator_lead2zero_preprocessed, generator_lead2zero
from hyperband.hyperband_base_model import CustomHistory
from sklearn.utils import class_weight
import itertools

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


class EnsembleModel:
    def __init__(self, id, learning_rate, epsilon, preprocessing_func = None, preprocessing_args = {}, lead_configuration=None, model_path="ensemble_model", magic_vector_path="resources/magic_vector.npy"):
        self.model_path = model_path
        self.magic_vector_path = magic_vector_path
        self.magic_vector = np.load(magic_vector_path)
        self.lead_configuration = lead_configuration if lead_configuration is not None else [
            (12,)]
        self.models = []
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.id = id
        self.preprocessing = preprocessing_func
        self.preprocessing_args = preprocessing_args

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

    def fit(self, X_pointer, y_pointer, train_idx, val_idx, epochs, batch_size=64, preprocessing = None, preprocessing_args={}):
        if preprocessing is None:
            preprocessing = self.preprocessing
            preprocessing_args = self.preprocessing_args
        train_sample_count = len(train_idx)
        val_sample_count = len(val_idx)
        custom_h = CustomHistory()
        for combination in self.lead_configuration:
            print(
                f'Train combination - {"-".join(map(str, combination))}')
            h = None
            model_checkpoint = ModelCheckpoint(
                filepath=f'{self.model_path}/combination_lead_{"-".join(map(str, combination))}_fold{self.id}/cp.ckpt',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_best_only=True)
            early_stopping_callback = EarlyStopping(
                monitor='val_loss', patience=15, verbose=1, mode='min')
            model = get_compiled_architecture(
                self.learning_rate, self.epsilon)
            train_generator = None
            val_generator = None
            if preprocessing is not None:
                train_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, preprocessing, preprocessing_args, train_idx, True, X_pointer, y_pointer)
                val_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, preprocessing, preprocessing_args, val_idx, False,  X_pointer, y_pointer)
            else:
                train_generator = generator_lead2zero(
                    list(combination), batch_size, train_idx, True, X_pointer, y_pointer)
                val_generator = generator_lead2zero(
                    list(combination), batch_size, val_idx, False, X_pointer, y_pointer)

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
            p = Path(f'{self.model_path}/combination_lead_{"-".join(map(str, combination))}_fold{self.id}/history')
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(f'{self.model_path}/combination_lead_{"-".join(map(str, combination))}_fold{self.id}/history', "wb") as file_pi:
                pickle.dump(h.history, file_pi)
        return custom_h

    def load_model(self, models_path=None):
        path = models_path if models_path is not None else self.model_path
        for combination in self.lead_configuration:
            model = get_reference_architecture()
            path_to_file = f'{path}/combination_lead_{"-".join(map(str, combination))}_fold{self.id}/cp.ckpt'
            checkpoint_path = Path(path_to_file+".index")
            if not checkpoint_path.is_file():
                return False
            else:
                model.load_weights(path_to_file).expect_partial()
                model = compile_architecture(model, self.learning_rate, self.epsilon)
                self.models.append(model)
        return True

    def predict(self, X_pointer, test_idx, batch_size=64, preprocessing=None, preprocessing_args={}):
        if len(self.models) == 0:
            if not self.load_model():
                return None
        sample_count = len(test_idx)
        y_preds = []
        for i, combination in enumerate(self.lead_configuration):
            print(f'Predict combination - {"-".join(map(str, combination))}')
            test_generator = None
            if preprocessing is not None:
                test_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, test_idx, False, preprocessing, preprocessing_args, X_pointer)
            else:
                test_generator = generator_lead2zero(
                    list(combination), batch_size, test_idx, False, X_pointer)

            local_y_preds = self.models[i].predict(
                test_generator, batch_size=batch_size, steps=math.ceil(sample_count / batch_size))
            y_preds.append(local_y_preds)
            del local_y_preds

        y_preds = np.array(y_preds)
        y_preds = np.mean(y_preds, axis=0)
        # y_preds = y_preds * self.magic_vector
        y_preds = self.preds2onehot(y_preds)
        return y_preds
