import pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import h2o
from blender import blend_predictions
import json

#blend_train, blend_test = pickle.load(open('output/baseline_lvl1.pickle'))
#np.savetxt('output/baseline_lvl1_train_20.csv', blend_train, delimiter=',')
#np.savetxt('output/baseline_lvl1_test_20.csv', blend_test, delimiter=',')

h2o.init(max_mem_size=4)

train = h2o.import_file('output/baseline_lvl1_train_20.csv', destination_frame='train')
target = h2o.import_file('output/y.csv', destination_frame='target')
train['target'] = target.asfactor()
test = h2o.import_file('output/baseline_lvl1_test_20.csv', destination_frame='test')

# define feature and response columns
y='target'
x=list(set(train.col_names).difference(y))

# create and train model
clf = h2o.H2ODeepLearningEstimator(
    model_id='dl_1',
    nfolds=5,

    keep_cross_validation_predictions=True,
    distribution='bernoulli',
    hidden= [200,200,200],
    epochs=50
)
    #activation='TanhWithDropout')
clf.train(x=x, y=y, training_frame=train)
dl_err = clf.logloss(xval=True)
print 'cv-logloss:', dl_err

#"Tanh", "TanhWithDropout", "Rectifier", "RectifierWithDropout",
#"Maxout", or "MaxoutWithDropout"

#0.448034944768 - default
#0.45253988234 - RectifierWithDropout
#0.45323340921 - MaxoutWithDropout
#0.459647943448 - TanhWithDropout

# get holdout predictions
dl_train_scores = clf.cross_validation_holdout_predictions().as_data_frame()
dl_train_scores = dl_train_scores['p1']

# score test data
dl_test_scores = clf.predict(test)
dl_test_scores = dl_test_scores.as_data_frame()
dl_test_scores = dl_test_scores['p1']

blend_train, blend_test = pickle.load(open('output/baseline_lvl2_20.pickle'))

#dl_train = np.genfromtxt('output/deep_learning_pred_train.csv', delimiter=',')
#dl_test = np.genfromtxt('output/deep_learning_pred_test.csv', delimiter=',')

new_blend_train = np.column_stack([blend_train, dl_train_scores])
new_blend_test = np.column_stack([blend_test, dl_test_scores])

baseline_lvl2_dl = new_blend_train, new_blend_test
pickle.dump(baseline_lvl2_dl, open("output/baseline_lvl2_dl.pickle", "w"), protocol=2)

# eval updated blended results
random_seed = 1337
train_preds, test_preds = pickle.load(open('output/baseline_lvl2_dl.pickle'))
labels = pickle.load(open('output/y.pickle'))
ids_sub = pickle.load(open('output/ids_submission.pickle'))
skf = list(StratifiedKFold(labels, n_folds=20, random_state=random_seed))
err = blend_predictions(train_preds, test_preds, labels, ids_sub, skf, save_results=True)

with open('dominostats.json', 'wb') as f:
    f.write(json.dumps({"blended-logloss": err, "dl-logloss": dl_err}))

h2o.shutdown(prompt=False)



#0.446476731607