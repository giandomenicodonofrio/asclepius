import numpy as np
from pathlib import Path

def dehot(y):
    return np.argmax(y)

def custom_score(y_preds, y_true):
    A = np.zeros((9,9))
    for y_p, y_t in zip(y_preds, y_true):
        A[dehot(y_t)][dehot(y_p)] += 1
    return A

def calc_challenge_score(y_preds, y_true, score_output_path="score.txt"):
    A = custom_score(y_preds, y_true)
    
    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))

    F1 = (F11+F12+F13+F14+F15+F16+F17+F18+F19) / 9

    ## following is calculating scores for 4 types: AF, Block, Premature contraction, ST-segment change.

    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))

    # print(A)
    print('Total File Number: ', np.sum(A))

    print("F11: ", F11)
    print("F12: ", F12)
    print("F13: ", F13)
    print("F14: ", F14)
    print("F15: ", F15)
    print("F16: ", F16)
    print("F17: ", F17)
    print("F18: ", F18)
    print("F19: ", F19)
    print("F1: ", F1)

    print("Faf: ", Faf)
    print("Fblock: ", Fblock)
    print("Fpc: ", Fpc)
    print("Fst: ", Fst)
    p = Path(score_output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(score_output_path, 'w') as score_file:
        # print (A, file=score_file)
        print ('Total File Number: %d\n' %(np.sum(A)), file=score_file)
        print ('F11: %0.3f' %F11, file=score_file)
        print ('F12: %0.3f' %F12, file=score_file)
        print ('F13: %0.3f' %F13, file=score_file)
        print ('F14: %0.3f' %F14, file=score_file)
        print ('F15: %0.3f' %F15, file=score_file)
        print ('F16: %0.3f' %F16, file=score_file)
        print ('F17: %0.3f' %F17, file=score_file)
        print ('F18: %0.3f' %F18, file=score_file)
        print ('F19: %0.3f\n' %F19, file=score_file)
        print ('F1: %0.3f\n' %F1, file=score_file)
        print ('Faf: %0.3f' %Faf, file=score_file)
        print ('Fblock: %0.3f' %Fblock, file=score_file)
        print ('Fpc: %0.3f' %Fpc, file=score_file)
        print ('Fst: %0.3f' %Fst, file=score_file)

        score_file.close()

def calc_score(y_pred, y_true, ref_class):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for prediction, true in zip(y_pred, y_true):

        ref_class_ind = int(ref_class) - 1

        y_p = prediction[ref_class_ind]
        y_t = true[ref_class_ind]

        TP += (y_p == y_t and y_t == 1)
        FN += (y_p != y_t and y_t == 1)
        FP += (y_p != y_t and y_t == 0)
        TN += (y_p == y_t and y_t == 0)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TP / (TP+FN),
        "FPR": FP / (FP+TN),
        "F1": 2*TP/(2*TP + FP + FN)
    }


def one_hot(class_ind):
    ris = np.zeros(9)
    ris[class_ind] = 1
    return ris

def de_one_hot(y):
    return np.argmax(y)

def preds2onehot(y_preds):
    y_preds = np.argmax(y_preds, axis=1)
    y = []
    for y_p in y_preds:
        y.append(one_hot(y_p))
    del y_preds
    return np.array(y)
