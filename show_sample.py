import h5py

from src.utils.visualization import ECGPLOT


def main():
    i = 1
    with h5py.File('dataset', 'r') as a, h5py.File('preprocessed_dataset', 'r') as b:
        x1 = a['X'][i]
        x2 = b['X'][i]

        ECGPLOT("", ["normal", "preprocessed"], True, x1, x2)


if __name__ == "__main__":
    main()

    
