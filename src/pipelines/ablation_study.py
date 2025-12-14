import argparse
import h5py
import tensorflow as tf

from src.config import EXPERIMENTS, LEAD_CONFIGURATIONS, stringify_lead_configuration
from src.models.ensemble_model_v2 import EnsembleModel
from src.training.evaluation import evaluate
from src.training.training import train
from src.utils.preprocessing import get_default_preprocesser
from src.utils.data_augmentation import get_default_augmenter


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-dp", "--dataset-path", help="dataset path", default="dataset")
    parser.add_argument("-op", "--output-path", help="output path")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["evaluate", "train"],
        default="evaluate",
        help="run evaluation (default) or training loop",
    )
    args = parser.parse_args()
    config = vars(args)
    print(config)

    dataset_path = config["dataset_path"]
    mode = config["mode"]
    default_output_path = "ablation_study" if mode == "evaluate" else "ablation_study_parallel"
    output_path = config["output_path"] if config["output_path"] is not None else default_output_path

    augmenter = get_default_augmenter()
    preprocesser = get_default_preprocesser()
    runner = evaluate if mode == "evaluate" else train

    with h5py.File(dataset_path, 'r') as f:
        X_pointer = f["X"]
        y_pointer = f["y"]
        class_indexes = {}
        for i in range(1, 10):
            class_indexes[f'{i}'] = f[f'x/{i}']

        for i, experiment in enumerate(EXPERIMENTS):
            print(f'\n\n  ----------- {experiment} -----------\n\n')

            for lead_configuration in LEAD_CONFIGURATIONS:
                model_configuration = {
                    "lead_configuration": lead_configuration,
                    "model_path": f"{output_path}/{experiment}/{stringify_lead_configuration(lead_configuration)}",
                    "preprocesser": preprocesser if i in [1, 3] else None,
                    "augmenter": augmenter if i in [2, 3] else None,
                }

                output_dir = f'{output_path}/evaluate_res/{experiment}/{stringify_lead_configuration(lead_configuration)}'
                runner(
                    EnsembleModel,
                    model_configuration,
                    X_pointer,
                    y_pointer,
                    class_indexes,
                    10,
                    100,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    main()
