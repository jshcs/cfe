import numpy as np
import pickle

Bibtexlabels = {'author':0,'title':1,'journal':2,'year':3,'volume':4,'pages':5}
files = ['natbib', 'achemso' , 'ajhg', 'biochem', 'mit-chicago', 'siamplain' ,'spbasic' ,'bmc-mathphys' ]

held_out_files = ['sageH', 'tfnlm']

umass_path = {"train" : "umass_train_x_y.npz" ,
              "test" : "umass_test_x_y.npz",
              "val" : "umass_val_x_y.npz"
              }

def make_format(files):
    fnames = []
    for fname in files :
        z = 'final_'+fname+'_pairs'
        fnames.append(z)
    return fnames

def get_umass_x_y(outfile):
    with open(outfile,"rb") as outp:
        X = pickle.load(outp)
        Y = pickle.load(outp)
    return X,Y

def load_X_Y_from_npy(fnames):
    X , Y = [] , []
    pairs = np.load(fnames)
    for p in pairs :
        x = []
        y = []
        for item in p :
            a , b  = item[0] , item[1]
            x.append(a)
            if b == 'author':
                y.append('person')
            else :
                y.append(b)
        X.append(x)
        Y.append(y)
    return X , Y

def make_train_test_split(X,Y):
    num = np.shape(X)[0]
    trainNum = int(num*0.6)
    valNum = int(num*0.25)
    testNum = num - trainNum - valNum

    X_train = X[:trainNum]
    y_train = Y[:trainNum]
    X_val = X[trainNum : trainNum + valNum]
    Y_val = Y[trainNum : trainNum + valNum]
    X_test = X[trainNum + valNum:]
    Y_test = Y[trainNum + valNum:]
    return X_train , y_train , X_test , Y_test , X_val , Y_val

def make_held_out_pickles(files, pklfile):
    X_t = []
    Y_t = []
    for fname in files :
        X_T , Y_T = load_X_Y_from_npy(fname +'.npy')
        print 'held' , len(X_T), len(Y_T)
        X_t.extend(X_T)
        Y_t.extend(Y_T)

    with open(pklfile +'_test_x_y.pkl','wb') as inp :
        pickle.dump(X_t,inp)
        pickle.dump(Y_t,inp)

    print len(X_t)


def make_combined_pickles(formatted_fname , pklfile):
    X_TR , Y_TR , X_TE , Y_TE = [] , [] , [] , []
    X_VAL , Y_VAL = [] , []
    for fname in formatted_fname :
        X_T , Y_T = load_X_Y_from_npy(fname +'.npy')
        print fname
        X_train , y_train , X_test , Y_test , X_val , Y_val  \
             = make_train_test_split(X_T,Y_T)
        X_TR.extend(X_train)
        Y_TR.extend(y_train)
        X_TE.extend(X_test)
        Y_TE.extend(Y_test)
        X_VAL.extend(X_val)
        Y_VAL.extend(Y_val)
        print len(X_TR), len(Y_TR)
        print len(X_TE), len(Y_TE)
        print len(X_VAL) , len(Y_VAL)

    X_train_umass , Y_train_umass = get_umass_x_y(umass_path["train"])
    X_val_umass , Y_val_umass = get_umass_x_y(umass_path["val"])
    X_test_umass , Y_test_umass = get_umass_x_y(umass_path["test"])

    print len(X_train_umass), len(Y_train_umass)
    print len(X_val_umass), len(Y_val_umass)
    print len(X_test_umass) , len(Y_test_umass)

    prefix = 'umass'
    with open(prefix +'_train_x_y.pkl','wb') as inp :
        pickle.dump(X_train_umass,inp)
        pickle.dump(Y_train_umass,inp)

    with open(prefix +'_val_x_y.pkl','wb') as inp :
        pickle.dump(X_val_umass,inp)
        pickle.dump(Y_val_umass,inp)

    with open(prefix +'_test_x_y.pkl','wb') as inp :
        pickle.dump(X_test_umass,inp)
        pickle.dump(Y_test_umass,inp)

    X_TR.extend(X_train_umass)
    Y_TR.extend(Y_train_umass)
    X_TE.extend(X_test_umass)
    Y_TE.extend(Y_test_umass)
    X_VAL.extend(X_val_umass)
    Y_VAL.extend(Y_val_umass)

    print len(X_TR), len(Y_TR)
    print len(X_TE), len(Y_TE)
    print len(X_VAL) , len(Y_VAL)

    with open(pklfile +'_train_x_y.pkl','wb') as inp :
        pickle.dump(X_TR,inp)
        pickle.dump(Y_TR,inp)

    with open(pklfile +'_val_x_y.pkl','wb') as inp :
        pickle.dump(X_VAL,inp)
        pickle.dump(Y_VAL,inp)

    with open(pklfile +'_test_x_y.pkl','wb') as inp :
        pickle.dump(X_TE,inp)
        pickle.dump(Y_TE,inp)

formatted_fname = make_format(files)
#print formatted_fname
#make_combined_pickles(formatted_fname , 'combined')
make_held_out_pickles(held_out_files , 'unseen')

