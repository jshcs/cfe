import pickle
import numpy as np
from config import *

with open('../../data/we_pickles/umass_train.pkl', 'rb') as inp:
    X_train = pickle.load(inp)
    y_train = pickle.load(inp)

with open('../../data/we_pickles/umass_val.pkl', 'rb') as inp:
    X_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

with open('../../data/we_pickles/umass_test.pickle', 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)

print X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape,y_test.shape

#loading bobtex data
for style in styleFile:
    with open('../../data/we_pickles/'+style+'_train.pkl', 'rb') as inp:
        bibtex_X_train = pickle.load(inp)
        bibtex_y_train = pickle.load(inp)

    with open('../../data/we_pickles/'+style+'_val.pkl', 'rb') as inp:
        bibtex_X_valid = pickle.load(inp)
        bibtex_y_valid = pickle.load(inp)

    with open('../../data/we_pickles/'+style+'_test.pickle', 'rb') as inp:
        bibtex_X_test = pickle.load(inp)
        bibtex_y_test = pickle.load(inp)

    X_train = np.concatenate((X_train,bibtex_X_train),axis = 0)
    y_train = np.concatenate((y_train,bibtex_y_train),axis = 0)
    X_valid = np.concatenate((X_valid,bibtex_X_valid),axis = 0)
    y_valid = np.concatenate((y_valid,bibtex_y_valid),axis = 0)
    X_test = np.concatenate((X_test,bibtex_X_test),axis = 0)
    y_test = np.concatenate((y_test,bibtex_y_test),axis = 0)



print X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape,y_test.shape


np.save('../../data/we_npy/combined_X_train.npy',X_train,allow_pickle=False)
np.save('../../data/we_npy/combined_y_train.npy',y_train,allow_pickle=False)
np.save('../../data/we_npy/combined_X_valid.npy',X_valid,allow_pickle=False)
np.save('../../data/we_npy/combined_y_valid.npy',y_valid,allow_pickle=False)
np.save('../../data/we_npy/combined_X_test.npy',X_test,allow_pickle=False)
np.save('../../data/we_npy/combined_y_test.npy',y_test,allow_pickle=False)


X_train=np.load('../../data/we_npy/combined_X_train.npy')
y_train=np.load('../../data/we_npy/combined_y_train.npy')
X_valid=np.load('../../data/we_npy/combined_X_valid.npy')
y_valid=np.load('../../data/we_npy/combined_y_valid.npy')
X_test=np.load('../../data/we_npy/combined_X_test.npy')
y_test=np.load('../../data/we_npy/combined_y_test.npy')

print X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape,y_test.shape
