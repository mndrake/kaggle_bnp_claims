import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import json
from xgboost import XGBClassifier
from scipy.stats import hmean
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import time
from hyperopt import fmin, tpe, hp, STATUS_OK
import pickle
from blender import blend_predictions

def logloss(attempt, actual, epsilon=1.0e-15):
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return -np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


def clip_probabilities(values, epsilon=1.0e-15):
    return np.clip(values, epsilon, 1-epsilon)

def train_classifiers(x_vals, x_sub, y_vals, skf, clfs, scores_only=False):
    # train classifiers and return out of fold train
    # predictions and blended submission predictions
    blend_train = np.zeros((x_vals.shape[0], len(clfs)))
    blend_test = np.zeros((x_sub.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        blend_test_j = np.zeros((x_sub.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            X_train = x_vals[train]
            y_train = y_vals[train]
            X_test = x_vals[test]
            clf.fit(X_train, y_train)
            blend_train[test, j] = clf.predict_proba(X_test)[:,1]
            blend_test_j[:, i] = clf.predict_proba(x_sub)[:,1]
        blend_test[:,j] = hmean(clip_probabilities(blend_test_j), axis=1)
        print "clf:",j,"logloss:", logloss(y_vals,blend_train[:,j])

    dataset_blend_train = clip_probabilities(blend_train)
    dataset_blend_test = clip_probabilities(blend_test)
    if scores_only:
        return dataset_blend_train, dataset_blend_test
    else:
        X_stacked = np.hstack([x_vals, dataset_blend_train])
        X_submission_stacked = np.hstack([x_sub, dataset_blend_test])
        return X_stacked, X_submission_stacked


# def blend_predictions(blend_train, blend_test, y, ids_sub, skf, save_results=False):
#     clf = LogisticRegression()
#     prob_cv = np.zeros(blend_train.shape[0])
#
#     for i, (train, test) in enumerate(skf):
#         X_train = blend_train[train]
#         y_train = y[train]
#         X_test = blend_train[test]
#         clf.fit(X_train, y_train)
#         prob_cv[test] = clf.predict_proba(X_test)[:, 1]
#
#     # calculate CV evaluation metric
#     cv_logloss = logloss(y, prob_cv)
#
#     # blending submission
#     clf.fit(blend_train, y)
#     y_submission = clf.predict_proba(blend_test)[:, 1]
#
#     if save_results:
#         csv_output = 'submission_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))
#         pd.DataFrame({"ID": ids_sub, "PredictedProb": y_submission}).to_csv(csv_output, index=False)
#         print 'saving:', csv_output
#
#     # overall scoring metric
#     print 'logloss:', cv_logloss
#     return cv_logloss


def reprocess_classifier(x_vals, x_sub, prior_train, prior_test, y_vals, j, skf, clf, scores_only=False):
    # train classifiers and return out of fold train
    # predictions and blended submission predictions
    blend_train = prior_train
    blend_test = prior_test

    blend_test_j = np.zeros((x_sub.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        X_train = x_vals[train]
        y_train = y_vals[train]
        X_test = x_vals[test]
        clf.fit(X_train, y_train)
        blend_train[test, j] = clf.predict_proba(X_test)[:, 1]
        blend_test_j[:, i] = clf.predict_proba(x_sub)[:, 1]
    blend_test[:, j] = hmean(clip_probabilities(blend_test_j), axis=1)
    print "clf:", j, "logloss:", logloss(y_vals, blend_train[:, j])

    dataset_blend_train = clip_probabilities(blend_train)
    dataset_blend_test = clip_probabilities(blend_test)
    if scores_only:
        return dataset_blend_train, dataset_blend_test
    else:
        X_stacked = np.hstack([x_vals, dataset_blend_train])
        X_submission_stacked = np.hstack([x_sub, dataset_blend_test])
        return X_stacked, X_submission_stacked




# -------------------------------------------------------------------------

# Initialize Variables

random_seed = 1337

y = pickle.load(open('output/y.pickle'))
ids_submission = pickle.load(open('output/ids_submission.pickle'))

baseline_lvl1 = pickle.load(open('output/baseline_lvl1.pickle'))
baseline_lvl2 = pickle.load(open('output/baseline_lvl2.pickle'))

n_threads = -1  #32
skf = list(StratifiedKFold(y, n_folds = 10, random_state=random_seed))

## baseline classifiers

clfs1 = [RandomForestClassifier(n_estimators=250, n_jobs=n_threads, criterion='gini', random_state=random_seed, min_samples_leaf=4),
         RandomForestClassifier(n_estimators=450, n_jobs=n_threads, criterion='entropy', random_state=random_seed, min_samples_leaf=2),
         ExtraTreesClassifier(n_estimators=300, n_jobs=n_threads, criterion='gini', random_state=random_seed),
         ExtraTreesClassifier(n_estimators=475, n_jobs=n_threads, criterion='entropy', random_state=random_seed),
         XGBClassifier(learning_rate=0.05, n_estimators=200, objective="binary:logistic", nthread=n_threads, seed=random_seed)]

clfs2 = [RandomForestClassifier(n_estimators=200, n_jobs=n_threads, criterion='gini', random_state=random_seed,
                                min_samples_leaf=4, max_features=0.55, max_depth=3),
         RandomForestClassifier(n_estimators=100, n_jobs=n_threads, criterion='entropy', random_state=random_seed,
                                max_features=0.9, max_depth=12, min_samples_leaf=3),
         ExtraTreesClassifier(n_estimators=150, n_jobs=n_threads, criterion='gini', random_state=random_seed),
         ExtraTreesClassifier(n_estimators=300, n_jobs=n_threads, criterion='entropy', random_state=random_seed),
         XGBClassifier(learning_rate=0.05, n_estimators=100, objective="binary:logistic", nthread=n_threads,
                       seed=random_seed, colsample_bytree=0.9, min_child_weight=4, subsample=1, max_depth=9,
                       gamma=0.65)]

#{'colsample_bytree': 0.8500000000000001, 'min_child_weight': 1.0, 'subsample': 0.9500000000000001,
# 'eta': 0.05, 'max_depth': 14.0, 'gamma': 0.9}

print 'base score'
base_score = blend_predictions(baseline_lvl2[0], baseline_lvl2[1], y, ids_submission, skf, save_results=False)


# Define the hyperparameter space

tree_space = dict(
    max_features = hp.quniform('max_features', 0.5, 1, 0.05),
    max_depth = hp.quniform('max_depth', 1, 15, 1),
    min_samples_leaf = hp.quniform('min_samples_leaf', 1, 6, 1))


xgb_space = {'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 15, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'eval_metric': 'logloss',
             'objective': 'binary:logistic',
             'silent': 1}

def xgb_objective(params):
    print "training model with parameters: "
    print params
    new_clf = XGBClassifier(
        max_depth=int(params['max_depth']),
        learning_rate=params['eta'],
        n_estimators=200,
        objective=params['objective'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        gamma=params['gamma'],
        colsample_bytree=params['colsample_bytree'],
        silent=params['silent'])
    new_lvl2 = reprocess_classifier(
        x_vals=baseline_lvl1[0],
        x_sub=baseline_lvl1[1],
        prior_train=baseline_lvl2[0],
        prior_test=baseline_lvl2[1],
        y_vals=y, j=4,
        skf=skf, clf=new_clf, scores_only=True)
    new_score = blend_predictions(new_lvl2[0], new_lvl2[1], y, ids_submission, skf, save_results=False)
    print "Score {0}\n\n".format(new_score)
    return {'loss': new_score, 'status': STATUS_OK}

def rf2_objective(params):
    print "training model with parameters: "
    print params
    new_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=n_threads,
        criterion='entropy',
        random_state=random_seed,
        min_samples_leaf=params['min_samples_leaf'],
        max_depth=params['max_depth'],
        max_features=params['max_features'])
    new_lvl2 = reprocess_classifier(
        x_vals=baseline_lvl1[0],
        x_sub=baseline_lvl1[1],
        prior_train=baseline_lvl2[0],
        prior_test=baseline_lvl2[1],
        y_vals=y, j=1,
        skf=skf, clf=new_clf, scores_only=True)
    new_score = blend_predictions(new_lvl2[0], new_lvl2[1], y, ids_submission, skf, save_results=False)
    #print "Score {0}\n\n".format(new_score)
    return {'loss': new_score, 'status': STATUS_OK}

def et1_objective(params):
    print "training model with parameters: "
    print params
    new_clf = ExtraTreesClassifier(
        n_estimators=150,
        n_jobs=n_threads,
        criterion='gini',
        random_state=random_seed,
        min_samples_leaf=params['min_samples_leaf'],
        max_depth=params['max_depth'],
        max_features=params['max_features'])
    new_lvl2 = reprocess_classifier(
        x_vals=baseline_lvl1[0],
        x_sub=baseline_lvl1[1],
        prior_train=baseline_lvl2[0],
        prior_test=baseline_lvl2[1],
        y_vals=y, j=2,
        skf=skf, clf=new_clf, scores_only=True)
    new_score = blend_predictions(new_lvl2[0], new_lvl2[1], y, ids_submission, skf, save_results=False)
    #print "Score {0}\n\n".format(new_score)
    return {'loss': new_score, 'status': STATUS_OK}

def et2_objective(params):
    print "training model with parameters: "
    print params
    new_clf = ExtraTreesClassifier(
        n_estimators=300,
        n_jobs=n_threads,
        criterion='entropy',
        random_state=random_seed,
        min_samples_leaf=params['min_samples_leaf'],
        max_depth=params['max_depth'],
        max_features=params['max_features'])
    new_lvl2 = reprocess_classifier(
        x_vals=baseline_lvl1[0],
        x_sub=baseline_lvl1[1],
        prior_train=baseline_lvl2[0],
        prior_test=baseline_lvl2[1],
        y_vals=y, j=3,
        skf=skf, clf=new_clf, scores_only=True)
    new_score = blend_predictions(new_lvl2[0], new_lvl2[1], y, ids_submission, skf, save_results=False)
    #print "Score {0}\n\n".format(new_score)
    return {'loss': new_score, 'status': STATUS_OK}


# Evaluate the function fmin over the hyperparameter space, and
# print the best hyperparameters.
best = fmin(xgb_objective, space=xgb_space, algo=tpe.suggest, max_evals=60)
#best = fmin(et2_objective, space=tree_space, algo=tpe.suggest, max_evals=60)
print "Optimal parameters for dtrain are: ", best