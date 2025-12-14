from keras.callbacks import ModelCheckpoint, EarlyStopping
import math
import pickle
import numpy as np
import gc
import tensorflow as tf
from keras import backend as K
from pathlib import Path
from reference.architecture import get_compiled_architecture
from src.utils.data import generator_lead2zero_preprocessed, generator_lead2zero
from hyperband.hyperband_base_model import HyperBandBaseModel, CustomHistory
from src.utils.utility import path_creator


class EnsembleHyperModel(HyperBandBaseModel):
    def __init__(self, conf, conf_idx, fold_idx, preprocessing = None, model_path="reference_ensemble_hypermodel", magic_vector_path="resources/magic_vector.npy"):
        self.conf = conf
        self.conf_idx = conf_idx
        self.fold_idx = fold_idx
        self.model_path = model_path
        self.magic_vector_path = magic_vector_path
        self.fit_num = 0
        self.preprocessing = preprocessing
        self.lead_configuration = conf["lead_combination"] if conf["lead_combination"] is not None else [(12,)]
        self.models = []

    def _get_conf_name(self):
        conf_name = ""
        for k, v in self.conf.items():
            if k not in ["lead_combination", "preprocessing_args"]:
                conf_name += str(v) + "_"
        return conf_name[:-1]

    def _get_model_saved_path(self, base_path=None):
        return f'{self.model_path if base_path is None else base_path}/fold{self.fold_idx}/{self._get_conf_name()}'

    def _get_sub_model_saved_path(self, lead_combination, base_path=None):
        return self._get_model_saved_path(base_path) + f'/lead_combination_{"-".join(map(str, lead_combination))}'

    def fit(self, X_pointer, y_pointer, train_idx, val_idx, epochs, prev_epochs, batch_size):
        train_sample_count = len(train_idx)
        val_sample_count = len(val_idx)
        custom_h = CustomHistory()
        for combination in self.lead_configuration:
            print(f'Train combination - {"-".join(map(str, combination))}')
            h = None
            conf_name = self._get_conf_name()
            model_checkpoint = ModelCheckpoint(
                filepath=self._get_sub_model_saved_path(combination) + "/cp.ckpt",
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
            early_stopping_callback = EarlyStopping(
                monitor='val_loss', patience=5, verbose=1, mode='min')
            model = get_compiled_architecture(self.conf["learning_rate"], self.conf["epsilon"])
            train_generator = None
            val_generator = None
            if self.preprocessing is not None:
                train_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, self.preprocessing, self.conf["preprocessing_args"], train_idx, True, X_pointer, y_pointer)
                val_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, self.preprocessing, self.conf["preprocessing_args"], val_idx, False,  X_pointer, y_pointer)
            else:
                train_generator = generator_lead2zero(
                    list(combination), batch_size, train_idx,True, X_pointer, y_pointer)
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
                                     model_checkpoint],
                          max_queue_size=16,
                          initial_epoch=prev_epochs)
            
            symbolic_weights = getattr(model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            optimizer_path = f'{self._get_sub_model_saved_path(combination)}/optimizer.pkl'
            path_creator(optimizer_path)
            with open(optimizer_path, 'wb') as f:
                pickle.dump(weight_values, f)

            max_val_acc_idx = np.argmax(np.array(h.history["val_accuracy"]))
            custom_h.history["accuracy"].append(h.history["accuracy"][max_val_acc_idx])
            custom_h.history["loss"].append(h.history["loss"][max_val_acc_idx])
            custom_h.history["val_accuracy"].append(h.history["val_accuracy"][max_val_acc_idx])
            custom_h.history["val_loss"].append(h.history["val_loss"][max_val_acc_idx])
            del model
            gc.collect()
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            history_path = f'{self._get_sub_model_saved_path(combination)}/history'
            path_creator(history_path)
            with open(history_path, "wb") as file_pi:
                pickle.dump(h.history, file_pi)
        self.fit_num += 1
        return custom_h

    def load_model(self, models_path=None):
        for combination in self.lead_configuration:
            model = get_compiled_architecture(self.conf["learning_rate"], self.conf["epsilon"])
            path_to_file = self._get_sub_model_saved_path(combination, base_path=models_path)
            checkpoint_path = Path(path_to_file+".index")
            if not checkpoint_path.is_file():
                return False
            else:
                model.load_model(path_to_file)
                model._make_train_function()
                with open(f'{path_to_file}/optimizer.pkl', 'rb') as f:
                    weight_values = pickle.load(f)
                model.optimizer.set_weights(weight_values)
                self.models.append(model)
        return True

    def predict(self, X_pointer, test_idx, batch_size):
        if len(self.models) == 0:
            if not self.load_model():
                return None
        sample_count = len(test_idx)
        y_preds = []
        for i, combination in enumerate(self.lead_configuration):
            print(f'Predict combination - {"-".join(map(str, combination))}')
            test_generator = None
            if self.preprocessing is not None:
                test_generator = generator_lead2zero_preprocessed(
                    list(combination), batch_size, test_idx, False, self.preprocessing, self.conf["preprocessing_args"], X_pointer)
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
        y_preds = super().preds2onehot(y_preds)
        return y_preds
