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

grid = h2o.H2OGridSearch()

# create and train model
clf = h2o.H2ODeepLearningEstimator(
    model_id='dl_1',
    nfolds=5,

    keep_cross_validation_predictions=True,
    distribution='bernoulli',
    hidden=[200, 200, 200],
    epochs=50
)

clf.train(x=x, y=y, training_frame=train)
dl_err = clf.logloss(xval=True)
print 'cv-logloss:', dl_err

# get holdout predictions
dl_train_scores = clf.cross_validation_holdout_predictions().as_data_frame()
dl_train_scores = dl_train_scores['p1']

blend_train, _ = pickle.load(open('output/baseline_lvl2_20.pickle'))

new_blend_train = np.column_stack([blend_train, dl_train_scores])

# eval updated blended results
random_seed = 1337
train_preds, _ = pickle.load(open('output/baseline_lvl2_dl.pickle'))
labels = pickle.load(open('output/y.pickle'))
skf = list(StratifiedKFold(labels, n_folds=20, random_state=random_seed))
err = blend_predictions(train_preds, None, labels, None, skf, save_results=False)

with open('dominostats.json', 'wb') as f:
    f.write(json.dumps({"blended-logloss": err, "dl-logloss": dl_err}))

h2o.shutdown(prompt=False)



# 0.44644