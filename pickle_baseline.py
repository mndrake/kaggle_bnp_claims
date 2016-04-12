import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import hmean
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import time
import pickle


def logloss(attempt, actual, epsilon=1.0e-15):
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return -np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


def clip_probabilities(values, epsilon=1.0e-15):
    return np.clip(values, epsilon, 1-epsilon)


def load_data():
    # read in data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    # combining to clean data
    combined = train.append(test)
    # drop variables
    combined = combined.drop(['v8','v23','v25','v31','v36','v37','v46','v51',
                             'v53','v54','v63','v73','v75','v79','v81','v82',
                             'v89','v92','v95','v105','v107','v108','v109',
                             'v110','v116','v117','v118','v119','v123','v124',
                             'v128'],axis=1)

    # create any new variables
    # factorize categorical variables
    for col in combined.columns:
        if combined[col].dtype == 'O':
            combined[col] = pd.factorize(combined[col])[0]
        else:
            tmp_len = len(combined[combined[col].isnull()])
            if tmp_len>0:
                combined.loc[combined[col].isnull(), col] = -999
    # spliting into train, test frames
    train = combined[:len(train)]
    test = combined[len(train):]

    feature_names = combined.columns.drop(['ID','target'])
    X = train[feature_names].values
    y = train['target'].values
    X_submission = test[feature_names].values
    ids_submission = test['ID'].values

    return X, y, X_submission, ids_submission


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


def blend_predictions(blend_train, blend_test, y, ids_sub, skf, save_results=False):
    clf = LogisticRegression()
    prob_cv = np.zeros(blend_train.shape[0])

    for i, (train, test) in enumerate(skf):
        X_train = blend_train[train]
        y_train = y[train]
        X_test = blend_train[test]
        clf.fit(X_train, y_train)
        prob_cv[test] = clf.predict_proba(X_test)[:, 1]

    # calculate CV evaluation metric
    cv_logloss = logloss(y, prob_cv)

    # blending submission
    clf.fit(blend_train, y)
    y_submission = clf.predict_proba(blend_test)[:, 1]

    if save_results:
        csv_output = 'submission_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))
        pd.DataFrame({"ID": ids_sub, "PredictedProb": y_submission}).to_csv(csv_output, index=False)
        print 'saving:', csv_output

    # overall scoring metric
    print 'logloss:', cv_logloss
    return cv_logloss


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

X, y, X_submission, ids_submission = load_data()
n_threads = -1  #32
skf = list(StratifiedKFold(y, n_folds = 20, random_state=random_seed))

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
                       seed=random_seed, colsample_bytree=0.85, min_child_weight=1, subsample=0.95, max_depth=14, gamma=0.9)]

#baseline_lvl1 = train_classifiers(X, X_submission, y, skf, clfs1)
baseline_lvl1 = pickle.load(open('output/baseline_lvl1_20.pickle'))
baseline_lvl2 = train_classifiers(baseline_lvl1[0], baseline_lvl1[1], y, skf, clfs2, scores_only=True)
base_score = blend_predictions(baseline_lvl2[0], baseline_lvl2[1], y, ids_submission, skf, save_results=False)

## pickle baseline levels

#pickle.dump(baseline_lvl1, open("results/baseline_lvl1_20.pickle", "w"), protocol=2)
pickle.dump(baseline_lvl2, open("output/baseline_lvl2_20.pickle", "w"), protocol=2)