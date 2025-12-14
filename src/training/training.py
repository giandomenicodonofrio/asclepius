from multiprocessing import Pool
from src.utils.data import dataset_train_val_test_division

class ConditionalParallelTrainer:
    def __init__(self, train_function, parallel_training: bool = False, n_workers: int = 3):
        self.parallel_training = parallel_training
        self.n_workers = n_workers
        self.train_function = train_function

    def __call__(self, *args, **kw):
        if self.parallel_training:
            with Pool(self.n_workers) as p:
                p.starmap(self.train_function, *args, **kw)
        else:
            for arg in args:
                self.train_function(*arg, **kw)



def train(modelClass, model_configuration, X_pointer, y_pointer, data_idx_by_class, k=10, epochs = 1, output_dir="evaluate_res"):
    dataset = dataset_train_val_test_division(data_idx_by_class, k)
    for i, (training_idx, val_idx, _) in enumerate(dataset):
        model = modelClass(i, **model_configuration)
        print(f'\n\n ----- Fold {i} ----- \n\n')
        model.fit(X_pointer, y_pointer, training_idx, val_idx, epochs)
