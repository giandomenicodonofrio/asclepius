import argparse
from collections import defaultdict
import h5py
from hyperband.ensemble_hypermodel import EnsembleHyperModel
from hyperband.hyperband_cv import HyperBandCV

from src.utils.preprocessing import butterworth_filter, zscore_normalization
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

learning_rates = [1e-2, 1e-3]
adam_epsilons = [1e-0, 1e-1]
lowcut=[0.05 * 3.0, 0.05 * 3.3]
highcut=[15, 20, 10]
fs=[200, 100]

mean=[0]
dev = [0.199]

lead_combinations = [[(12, ), (0, ), (1, )]]
"""lead_combinations = [
    [(12, )] ,
    [(12, ), (0, ), (1, ), (2, ), (3, ), (4, ), (5, ), (6, ), (7, ), (8, ), (9, ), (10, ), (11, )],
    [(12, ), (0, 1), (11, 6), (10, 8), (2, 7), (2, 9), (6, 8)]
]"""

param_map = {
    "learning_rate": learning_rates,
    "epsilon": adam_epsilons, 
    "lowcut": lowcut, 
    "highcut": highcut, 
    "fs": fs, 
    "dev": dev, 
    "mean": mean, 
    "lead_combination": lead_combinations
}

optimizer_params = {
    "learning_rate": learning_rates,
    "epsilon": adam_epsilons, 
}

preprocessing_params = {
    "lowcut": lowcut, 
    "highcut": highcut, 
    "fs": fs, 
    "dev": dev, 
    "mean": mean, 
}

ensemble_params = {
    "lead_combination": lead_combinations
}

# param_map = {**optimizer_params, **preprocessing_params, **ensemble_params}

""" preprocessing_args_keys = ["lowcut", "highcut", "fs", "dev", "mean"] """

""" def preprocessing_func(data, lowcut, highcut, fs, dev, mean):
    zscore_normalization(data, mean, dev)
    return butterworth_filter(data, lowcut, highcut, fs) """

def conf_name(conf):
    conf_name = ""
    for k, v in conf.items():
        if k not in ["lead_combination", "preprocessing_args"]:
            conf_name += str(v) + "_"
    return conf_name[:-1]

def main():
    parser = argparse.ArgumentParser(
        description="Hyperband parameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-dp",
        "--dataset-path",
        help="dataset path",
        default="dataset_72000_onehot_multiclass_id_fold",
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path

    with h5py.File(dataset_path, 'r') as f:
        X_pointer = f["X"]
        y_pointer = f["y"]
        class_indexes = defaultdict(list)
        for i in range(1, 10):
            class_indexes[f'{i}'] = f[f'x/{i}']
        for i in range(len(lead_combinations)):
            param_map = {**optimizer_params, **{"lead_combination": [lead_combinations[i]]}}
            tuner = HyperBandCV(
                EnsembleHyperModel,
                param_map,
                preprocessing=None,
                preprocessing_args_keys=[],
                max_epochs=100,
                k=2,
                factor=2,
            )

            conf = tuner.fit(X_pointer, y_pointer, class_indexes)
            print(f'\n\nbest_conf {lead_combinations[i]}={conf}\n\n---conf_name = {conf_name(conf)}---')


if __name__ == "__main__":
    main()
