import numpy as np
from scipy.optimize import fmin_cobyla
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import pickle
import pandas as pd
import time


def blended(c, x):
    result = None
    for i in range(len(c)):
        result = result + c[i] * x[i] if result is not None else c[i] * x[i]
    result /= sum(c)
    return result


def error(p, x, y):
    preds = blended(p, x)
    preds = np.clip(preds, 1e-15, 1-1e-15)
    err = log_loss(y, preds)
    return err


def constraint(p, *args):
    return min(p) - .0


def blend_predictions(train_preds, test_preds, labels, ids_sub, skf, save_results=False):
    test_index = None
    for _, test_idx in skf:
        test_index = np.append(test_index, test_idx) if test_index is not None else test_idx
    val_labels = labels[test_index]

    val_predictions, val_submission = [], []

    for i in range(np.shape(train_preds)[1]):
        val_predictions.append(train_preds[:,i])

    for i in range(np.shape(test_preds)[1]):
        val_submission.append(test_preds[:, i])

    p0 = [1.] * len(val_predictions)

    p = fmin_cobyla(error, p0, args=(val_predictions, val_labels), cons=[constraint], rhoend=1e-5)

    err = error(p, val_predictions, val_labels)
    print 'error:', err

    y_submission = blended(p, val_submission)

    if save_results:
        csv_output = 'submission_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))
        pd.DataFrame({"ID": ids_sub, "PredictedProb": y_submission}).to_csv(csv_output, index=False)
        print 'saving:', csv_output

    return err


if __name__ == '__main__':
    random_seed = 1337
    train_preds, test_preds = pickle.load(open('output/baseline_lvl2_dl.pickle'))
    labels = pickle.load(open('output/y.pickle'))
    ids_sub = pickle.load(open('output/ids_submission.pickle'))
    skf = list(StratifiedKFold(labels, n_folds=20, random_state=random_seed))
    blend_predictions(train_preds, test_preds, labels, ids_sub, skf, save_results=True)

# 0.446589072032
