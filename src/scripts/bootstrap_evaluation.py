
import itertools
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import EXPERIMENTS, LEAD_CONFIGURATIONS, stringify_lead_configuration
from src.utils.score import calc_score
from src.utils.utility import path_creator


def bootstrap(test_idx, times):
    bootstrap_tests = []
    l = len(test_idx)
    for _ in range(times):
        idxs = random.choices(test_idx, k=l)
        idxs.sort()
        bootstrap_tests.append(idxs)
    return np.array(bootstrap_tests)


result_base_path = Path("ablation_study/evaluate_res")
# Use only experiments that have results on disk
experiments = [ex for ex in EXPERIMENTS if result_base_path.joinpath(ex).exists()]
configurations = [stringify_lead_configuration(cfg) for cfg in LEAD_CONFIGURATIONS]
architectures_name = [f"{ex}-{conf}" for ex in experiments for conf in configurations]


def get_y_results():
    tmp_path = result_base_path.joinpath("solo_lead").joinpath("(12,)")
    y_result = []
    for i in range(9):
        with open(tmp_path.joinpath(f"fold_{i}_y_test_y_preds.pickle"), 'rb') as fp:
            y_test, _ = pickle.load(fp)
            y_result.extend(y_test)
    y_result = np.array(y_result)
    return y_result


def get_model_results():
    results = defaultdict(list)

    for ex in experiments:
        print(f"---- {ex} ----")
        ex_path = result_base_path.joinpath(ex)
        for conf in tqdm(configurations):
            conf_path = ex_path.joinpath(conf)
            conf_result = []
            for i in range(9):
                with open(conf_path.joinpath(f"fold_{i}_y_test_y_preds.pickle"), 'rb') as fp:
                    _, y_preds = pickle.load(fp)
                    conf_result.extend(y_preds)
            results[f'{ex}-{conf}'] = np.array(conf_result)
    return results


def calculate_model_bootstrap_result(model_results, y_result, metric="F1", times=1000):
    bootstrap_tests_idxs = bootstrap(np.arange(len(y_result)), times)

    print("Bootstrap metrics calculation")

    model_bootstrap_result = defaultdict(list)
    for b_test_idxs in bootstrap_tests_idxs:
        for ex in experiments:
            ex_path = result_base_path.joinpath(ex)
            for conf in configurations:
                mean = 0
                for label in range(9):
                    score = calc_score(
                        model_results[f'{ex}-{conf}'][b_test_idxs], y_result[b_test_idxs], label)
                    mean += score[f"{metric}"]
                mean /= 9
                model_bootstrap_result[f'{ex}-{conf}'].append(mean)

    model_bootstrap_result_path = f"bootstrap_result/model_bootstrap_result_{metric}.pickle"
    path_creator(model_bootstrap_result_path)
    with open(model_bootstrap_result_path, 'wb') as f:
        pickle.dump(model_bootstrap_result, f, pickle.HIGHEST_PROTOCOL)
    return model_bootstrap_result


def load_model_bootstrap_result(metric="F1"):
    path = f"bootstrap_result/model_bootstrap_result_{metric}.pickle"
    with open(path, 'rb') as f:
        model_bootstrap_result = pickle.load(f)
        return model_bootstrap_result


def comparissons(model_bootstrap_result, architectures_name):
    comparissons_results = defaultdict(float)
    for m_i, m_j in itertools.permutations(architectures_name, 2):
        count = 0
        for y_mi, y_mj in zip(model_bootstrap_result[m_i], model_bootstrap_result[m_j]):
            if y_mi < y_mj:
                count+= 1
        comparissons_results[f'{m_i}_vs_{m_j}'] = count / len(model_bootstrap_result[m_i])
    return comparissons_results


def fixed_comparissons(model_bootstrap_result, experiments, configurations, fixed_ex, fixed_conf):
    comparissons_fixed_ex = defaultdict(float)
    comparissons_fixed_conf = defaultdict(float)
    for conf in configurations:
        for ex in [item for item in experiments if item != fixed_ex]:
            print(ex)
            count = 0
            m_i = f"{fixed_ex}-{conf}"
            m_j = f'{ex}-{conf}'
            for y_mi, y_mj in zip(model_bootstrap_result[m_i], model_bootstrap_result[m_j]):
                if y_mi > y_mj:
                    count+= 1
            comparissons_fixed_ex[f'{m_i}_vs_{m_j}'] = count / len(model_bootstrap_result[m_i])

    for ex in experiments:
        for conf in [item for item in configurations if item != fixed_conf]:
            count = 0
            m_i = f"{ex}-{fixed_conf}"
            m_j = f'{ex}-{conf}'
            for y_mi, y_mj in zip(model_bootstrap_result[m_i], model_bootstrap_result[m_j]):
                if y_mi > y_mj:
                    count+= 1
            comparissons_fixed_conf[f'{m_i}_vs_{m_j}'] = count / len(model_bootstrap_result[m_i])

    return comparissons_fixed_ex, comparissons_fixed_conf

    



# y_results = get_y_results()
# model_results = get_model_results()
# model_bootstrap_result = calculate_model_bootstrap_result(model_results, y_results, "FPR")
model_bootstrap_result = load_model_bootstrap_result("FPR")


comparissons_results = comparissons(model_bootstrap_result, architectures_name)

comparissons_fixed_ex, comparissons_fixed_conf = fixed_comparissons(model_bootstrap_result, experiments, configurations, "lead_preprocessing_augmentation", "(12,)-(0, 1, 6)-(10, 2, 8)-(11, 7, 9)-(1, 6, 8)")

print(json.dumps(comparissons_fixed_ex))
print("\n\n")
print(json.dumps(comparissons_fixed_conf))
