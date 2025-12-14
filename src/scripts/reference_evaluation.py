import argparse
import h5py
import tensorflow as tf

from src.models.ensemble_model import EnsembleModel
from src.training.evaluation import evaluate

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def main():
    parser = argparse.ArgumentParser(
        description="Reference ensemble evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-dp", "--dataset-path", help="dataset path", default="dataset")
    parser.add_argument("-op", "--output-path", help="output path", default="reference_model_evaluation")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    dataset_path = config["dataset_path"]
    output_path = config["output_path"]

    with h5py.File(dataset_path, 'r') as f:
        X_pointer = f["X"]
        y_pointer = f["y"]
        class_indexes = {}
        for i in range(1, 10):
            class_indexes[f'{i}'] = f[f'x/{i}']

        model_configuration = {
            "learning_rate": 0.001,  # ADAM default
            "epsilon": 1e-07,        # ADAM default
            "lead_configuration": [(12, ), (1, ), (2, ), (3, ), (4,), (5,), (6,), (7,), (8,), (9,), (11,)],
            "model_path": output_path,
        }

        evaluate(EnsembleModel, model_configuration, X_pointer, y_pointer, class_indexes, 10, 100)


if __name__ == "__main__":
    main()
