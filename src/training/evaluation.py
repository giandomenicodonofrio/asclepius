from src.utils.data import dataset_train_val_test_division
from src.utils.score import *
from src.utils.visualization import print_score_tot, print_fold_iteration_score
from collections import defaultdict
import pickle
from src.utils.utility import path_creator

def evaluate(modelClass, model_configuration, X_pointer, y_pointer, data_idx_by_class, k=10, epochs = 1, output_dir="evaluate_res"):
    dataset = dataset_train_val_test_division(data_idx_by_class, k)
    overall_scores = []
    models = []
    for i, (training_idx, val_idx, test_idx) in enumerate(dataset):
        model = modelClass(i, **model_configuration)
        print(f'\n\n ----- Fold {i} ----- \n\n')
        h = model.fit(X_pointer, y_pointer, training_idx, val_idx, epochs)
        if h is None:
            print(f'Ensemble already trained')
            with open(f"{output_dir}/my_thesis_scores/fold{i}_score.pickle", 'rb') as fp:
                fold_score = pickle.load(fp)
                print_fold_iteration_score(fold_score)
                overall_scores.append(fold_score)
            continue

        y_test = y_pointer[test_idx]
        y_preds = model.predict(X_pointer, test_idx)
        if y_preds is None:
            print("Problem with model, the model is not present")
            return
        fold_score = []
        print("calculate fold score ...")
        for ref_class in range(9):
            scores = calc_score(y_preds, y_test, ref_class)
            fold_score.append(scores)
        print("save fold score...")
        path_creator(f"{output_dir}/my_thesis_scores/fold{i}_score.pickle")
        with open(f"{output_dir}/my_thesis_scores/fold{i}_score.pickle", 'wb') as fp:
            pickle.dump(fold_score, fp, pickle.HIGHEST_PROTOCOL)

        overall_scores.append(fold_score)

        path_creator(f"{output_dir}/challenge_score/fold{i}_score.txt")
        calc_challenge_score(y_preds, y_test, f"{output_dir}/challenge_score/fold{i}_score.txt")

        path_creator(f'{output_dir}/fold_{i}_y_test_y_preds.pickle')
        with open(f'{output_dir}/fold_{i}_y_test_y_preds.pickle', 'wb') as fp:
            pickle.dump((y_test, y_preds), fp, pickle.HIGHEST_PROTOCOL)

        path_creator(f'{output_dir}/fold_{i}_score.pickle')
        with open(f'{output_dir}/fold_{i}_score.pickle', "wb") as fp:
            pickle.dump(fold_score, fp, pickle.HIGHEST_PROTOCOL)

        print_fold_iteration_score(fold_score)

    mean_scores = defaultdict(lambda: defaultdict(lambda : 0))
    for score in overall_scores:
        for i, sub_score in enumerate(score):
            for key in list(sub_score.keys()):
                mean_scores[f'{i}'][key] = round(mean_scores[f'{i}'][key] + sub_score[key], 4)

    for clss in list(mean_scores.keys()):
        for p_index in list(mean_scores[clss].keys()):
            mean_scores[clss][p_index] = round(mean_scores[clss][p_index] / len(overall_scores), 4)
    
    print_score_tot(mean_scores)
    return models, mean_scores
