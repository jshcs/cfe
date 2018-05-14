import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np

from config import *
import pickle

path_to_data = '../../data/feats_'

out_path = {'umass' : 'umass_y.pkl', 'comb': 'combined_y.pkl' ,'unseen' : 'unseen_y.pkl'}

input_unseen_path = {'test' : '../../data/unseen_test_x_y.pkl'}
input_path_umass = {"train" : "../../data/umass_train_x_y.pkl" ,
                   "test" : "../../data/umass_test_x_y.pkl",
                    "val" : "../../data/umass_val_x_y.pkl"
                   }

input_path_comb = {"train" : "../../data/combined_train_x_y.pkl" ,
                   "test" : "../../data/combined_test_x_y.pkl",
                    "val" : "../../data/combined_val_x_y.pkl"
                   }

def get_output_fname(input_path_key,out_path_key):
    fname  = path_to_data
    fname += input_path_key
    fname += '_'
    fname += out_path[out_path_key]
    return fname

def files_for_testcase13(umass_comb):
    fnames = {}
    for key in ['train','test', 'val'] :
        fname = get_output_fname(key,umass_comb)
        fnames[key] = fname
    return fnames

def files_for_testcase34(fnames):
    for key in ['test'] :
        fname = get_output_fname(key,'unseen')
        fnames[key] = fname
    return fnames

test_case_1 = files_for_testcase13('umass')
test_case_3 = files_for_testcase13('comb')
test_case_4 = files_for_testcase34(test_case_3)


def crf(path_dict,outfile,modelname):
    train_path = path_dict['train']
    val_path = path_dict['val']
    test_path = path_dict['test']

    with open(train_path, 'rb') as inp:
        X_train = pickle.load(inp)
        y_train = pickle.load(inp)

    with open(val_path, 'rb') as inp:
        X_val = pickle.load(inp)
        y_val = pickle.load(inp)

    with open(test_path, 'rb') as inp:
        X_test = pickle.load(inp)
        y_test = pickle.load(inp)

    labels = ['person' ,'title' , 'journal' ,'volume' ,'year' ,'pages']

    best_f1_score = -1
    c2_best = 0.0
    c1_best = 0.0

    for c1_val in [0.1 , 0.2 , 0.3] :
        for c2_val in [0.1 , 0.2 , 0.3] :
            crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True,
            c2 = c2_val,
            c1 = c1_val
            )
            crf.fit(X_train, y_train)
            y_pred = crf.predict(X_val)
            f1_score_curr = metrics.flat_f1_score(y_val,y_pred,average ='weighted')
            print f1_score_curr
            if f1_score_curr > best_f1_score :
                best_f1_score = f1_score_curr
                c2_best = c2_val
                c1_best = c1_val
                pickle.dump(crf, open(modelname, 'wb'))
                print 'best c2_best' ,c2_best , c1_best
                print(metrics.flat_classification_report(
            y_val, y_pred, labels=labels, digits=3
            ))
    #c1 = 0.111
    #c2 = 0.222
    print c1_best , c2_best
    crf = pickle.load(open(modelname,'rb'))
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
        ))
    np.savez(outfile,y_pred,y_test)


#dict_for_umass_unseen = files_for_testcase13('comb')
#crf(files_for_testcase34(dict_for_umass_unseen),'combT_U','combT_U')

#dict_for_umass_unseen = files_for_testcase13('umass')
#crf(files_for_testcase34(dict_for_umass_unseen),'umassT_U','umassT_U')

#crf(files_for_testcase13('umass'),'umassT_T','umassT_T')

#crf(files_for_testcase13('comb'),'combT_T','combT_T')