import numpy as np

# X_train=np.load('../../data/we_npy/combined_X_train.npy')
# print "done"
# y_train=np.load('../../data/we_npy/combined_y_train.npy')
# print "done"
# X_valid=np.load('../../data/we_npy/combined_X_valid.npy')
# print "done"
# y_valid=np.load('../../data/we_npy/combined_y_valid.npy')
# print "done"
# X_test=np.load('../../data/we_npy/combined_X_test.npy')
# print "done"
# y_test=np.load('../../data/we_npy/combined_y_test.npy')
# print "done"
#
#
# np.savez_compressed('../../data/we_npy/combined_dataset.npz',combined_X_train=X_train,combined_y_train=y_train,combined_X_valid=X_valid,combined_y_valid=y_valid,combined_X_test=X_test,combined_y_test=y_test)
a=np.load('../../data/we_npy_no_bio/combined_dataset.npz')
b=np.load('../../data/we_npy/combined_dataset.npz')
print a['combined_y_train'].shape,b['combined_y_train'].shape