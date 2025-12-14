import numpy as np

def generator(batch_size, indexes, leads=[12],  shuffle=True, preprocesser = None, augmenter=None, *args):
    inner_generator = _generation(batch_size, indexes, shuffle, *args)
    while True:
        x = next(inner_generator)
        batch = None
        batch, *_ = x
        set_lead2zero(batch, leads)
        if preprocesser is not None:
            batch = preprocesser(batch)
        if augmenter is not None:
            batch = augmenter(batch)
        yield x

def set_lead2zero(batch, lead_ids):
    if lead_ids != [12]:
        zeroIndices = np.asarray(list(set(range(12)) - set([*lead_ids])))
        batch[:, :, zeroIndices] = 0

def _generation(batch_size, indexes, shuffle=True, *args):
    sample_count = len(indexes)
    if shuffle:
        np.random.shuffle(indexes)
    while True:
        for idx in range(0, sample_count, batch_size):
            batch_idxs = indexes[idx: idx + batch_size]
            batch_idxs = np.sort(batch_idxs)
            yield tuple(data[batch_idxs] for data in args)
        
        if shuffle:
            np.random.shuffle(indexes)

def _generator(batch_size, indexes, shuffle=True, *args):
    inner_generator = _generation(batch_size, indexes, shuffle, *args)
    """ if len(args) == 1:
        while True:
            x = next(inner_generator)
            yield x[0] """
    yield from inner_generator


def generator_preprocessed(batch_size, indexes, preprocessing, preprocessing_args, shuffle=True, *args):
    inner_generator = _generation(batch_size, indexes, shuffle, *args)
    while True:
        x = next(inner_generator)
        batch = None
        batch, *_ = x
        for ecg in batch:
            preprocessing(ecg, **preprocessing_args)
        yield x

def print_tuple_shape(tuple):
    print("(", end="")
    for i, l in enumerate(tuple):
        print(len(l), end=", " if i < len(tuple)-1 else "")
    print(")")

def generator_lead2zero(lead_id, batch_size, indexes, shuffle=True, *args):
    inner_generator = generator(batch_size, indexes, shuffle, *args)
    while True:
        x = next(inner_generator)
        batch = None
        batch, *_ = x
        for ecg in batch:
            set2zero_lead(ecg, lead_id)
        yield x

def generator_lead2zero_preprocessed(lead_id, batch_size, preprocessing, preprocessing_args, indexes, shuffle=True, *args):
    inner_generator = generator(batch_size, indexes, shuffle, *args)
    while True:
        x = next(inner_generator)
        batch = None
        batch, *_ = x
        for ecg in batch:
            set2zero_lead(ecg, lead_id)
            preprocessing(ecg, **preprocessing_args)
        yield x


def set2zero_lead(X, lead_ids):
    if lead_ids != [12]:
        zeroIndices = np.asarray(list(set(range(12)) - set([*lead_ids])))
        X[:,zeroIndices] = 0


def nested_dataset_division(idx_by_class, k):
    dataset_idx = [{}]*k
    for i in range(k):
        for c in list(idx_by_class.keys()):
            l = len(idx_by_class[c]) // k
            dataset_idx[i][c]=idx_by_class[i * l: (i+1) * l]
    return dataset_idx

def flat_dataset_division(idx_by_class, k):
    dataset_idx = []
    for _ in range(k):
        val_idx = []
        for c in list(idx_by_class.keys()):
            d = np.array(idx_by_class[c])
            np.random.shuffle(d)
            l = len(d) // k
            choice = d[: l]
            val_idx.extend(choice)
        val_idx = np.sort(val_idx)
        training_idx = []
        training_idx = [index for l in list(
            idx_by_class.values()) for index in l if index not in val_idx]
        training_idx.sort()
        training_idx = np.array(training_idx)
        dataset_idx.append((training_idx, val_idx))
    return dataset_idx

def dataset_division(idx_by_class, k, flat=True):
    if flat:
        return flat_dataset_division(idx_by_class, k)
    else:
        return nested_dataset_division(idx_by_class, k)

def train_val_test_division(idx_by_class, k):
    dataset_idx = []
    for i in range(k - 1):
        val_idx = []
        test_idx = []
        for c in list(idx_by_class.keys()):
            d = np.array(idx_by_class[c])
            # np.random.shuffle(d)
            l = len(d) // k
            choice = d[i * l: (i+1) * l]
            val_idx.extend(choice)
            choice = d[(i+1)*l:(i+2)*l]
            test_idx.extend(choice)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)
        training_idx = []
        training_idx = [index for l in list(
            idx_by_class.values()) for index in l if index not in val_idx and index not in test_idx]
        training_idx.sort()
        training_idx = np.array(training_idx)
        dataset_idx.append((training_idx, val_idx, test_idx))
    return dataset_idx

def dataset_train_val_test_division(idx_by_class, k):
    return train_val_test_division(idx_by_class, k)
