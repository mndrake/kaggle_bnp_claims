import pickle
import numpy as np

blend_train, blend_test = pickle.load(open('output/baseline_lvl2_20.pickle'))

dl_train = np.genfromtxt('output/deep_learning_pred_train.csv', delimiter=',')
dl_test = np.genfromtxt('output/deep_learning_pred_test.csv', delimiter=',')

new_train = np.column_stack([blend_train, dl_train])
new_test = np.column_stack([blend_test, dl_test])

baseline_lvl2_dl = new_train, new_test

pickle.dump(baseline_lvl2_dl, open("output/baseline_lvl2_dl.pickle", "w"), protocol=2)