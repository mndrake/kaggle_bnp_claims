import pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import h2o
from blender import blend_predictions
import json
from h2o.grid.grid_search import H2OGridSearch

h2o.init(max_mem_size=4)

train = h2o.import_file('output/baseline_lvl1_train_20.csv', destination_frame='train')
target = h2o.import_file('output/y.csv', destination_frame='target')
train['target'] = target.asfactor()

# define feature and response columns
y = 'target'
x = list(set(train.col_names).difference(y))

search_criteria = {'strategy': 'RandomDiscrete',
                   'max_models': 60,
                   'max_runtime_secs': 28800,
                   'seed': 1773,
                   'stopping_rounds': 10,
                   'stopping_tolerance': 1e-4}

hyper_parameters = {
    'activation': ['Rectifier', 'RectifierWithDropout', 'Maxout', 'MaxoutWithDropout'],
    'hidden': [[25,25], [50,50], [100,100], [200,200], [500,500],
               [25,25,25], [50,50,50], [100,100,100], [200,200,200], [500,500,500]],
    'input_dropout_ratio': [0, 0.05, 0.10],
    'l1': [0, 1e-6, 1e-5, 1e-4],
    'l2': [0, 1e-6, 1e-5, 1e-4]
}

# create and train model
clf = h2o.H2ODeepLearningEstimator(
    nfolds=5,
    distribution='bernoulli',
    epochs=50,
    stopping_metric="logloss",
    stopping_tolerance=1e-4,
    stopping_rounds=10,
    max_w2=10
)

grid = H2OGridSearch(clf, hyper_parameters, grid_id='dl_grid_random', search_criteria=search_criteria)
grid.train(x=x, y=y, training_frame=train)

grid_search_results = grid.sort_by('logloss')
best_model_id = grid_search_results['Model Id'][0]
best_model = h2o.get_model(best_model_id)
print '-- best params --'
print 'activation:', best_model.params['activation']['actual']
print 'hidden:', best_model.params['hidden']['actual']
print 'input_dropout_ratio:', best_model.params['input_dropout_ratio']['actual']
print 'l1:', best_model.params['l1']['actual']
print 'l2:', best_model.params['l2']['actual']

#print best_model.params
err = best_model.logloss(xval=True)
print 'logloss:', err

# get holdout predictions
#dl_train_scores = best_model.cross_validation_holdout_predictions().as_data_frame()
#dl_train_scores = dl_train_scores['p1']

#blend_train, _ = pickle.load(open('output/baseline_lvl2_20.pickle'))

#new_blend_train = np.column_stack([blend_train, dl_train_scores])

# eval updated blended results
#random_seed = 1337
#train_preds, _ = pickle.load(open('output/baseline_lvl2_dl.pickle'))
#labels = pickle.load(open('output/y.pickle'))
#skf = list(StratifiedKFold(labels, n_folds=20, random_state=random_seed))
#err = blend_predictions(train_preds, None, labels, None, skf, save_results=False)

with open('dominostats.json', 'wb') as f:
    f.write(json.dumps({"dl-logloss": err}))

h2o.shutdown(prompt=False)

# 0.44644