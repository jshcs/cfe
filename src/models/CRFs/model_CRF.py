import sklearn_crfsuite
from sklearn_crfsuite import metrics
#from sklearn.metrics import  f1_score
import numpy as np

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1,dir_path +'/../')
#print sys.path

from config import *
import pickle

from CRF_features import *

def extract_features(ele_doc):
    feats = []
    for a,b in ele_doc :
        feat = CRF_Features(a)
        feat_str = feat.get_features()
        feats.append((feat_str))
    return feats

def pair(i, o):
    list_io = []
    for token,label in zip(i,o) :
        list_io.append((token,label))
    return list_io

def get_input(Ins,Outs) :
    docs = []
    for i , o in zip(Ins,Outs) :
        list_io =  pair(i,o)
        docs.append(list_io)
    return docs

def get_labels(doc):
    return [d for (a,d) in doc]


def crf():
    with open('../../../data/feats_train.pkl', 'rb') as inp:
        X_train = pickle.load(inp)
        y_train = pickle.load(inp)
    with open('../../../data/feats_val.pkl', 'rb') as inp:
        X_val = pickle.load(inp)
        y_val = pickle.load(inp)

    labels = ['title' , 'volume' ,'year' ,'journal','author','pages']

    with open('../../../data/feats_test.pkl', 'rb') as inp:
        X_test = pickle.load(inp)
        y_test = pickle.load(inp)

    best_f1_score = -1
    c2_best = 0.0
    c1_best = 0.0
    filename = 'crf_model.sav'

    '''

    for c1_val in np.linspace(0.0,1.0,num = 10) :
        for c2_val in np.linspace(0.0,1.0,num=10):
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
                pickle.dump(crf, open(filename, 'wb'))
                print 'best c2_best' ,c2_best , c1_best
                print(metrics.flat_classification_report(
            y_val, y_pred, labels=labels, digits=3
            ))
    '''
    c1 = 0.111
    c2 = 0.222
    crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True,
            c2 = c1,
            c1 = c2
            )
    print 'results :'
    crf = pickle.load(open(filename,'rb'))
    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
        ))


def save_data_pickle(fname1 , fname2 , path):
    Ins,Outs = [] , []
    with open(fname1, 'rb') as inp:
        X_train = pickle.load(inp)
        y_train = pickle.load(inp)
    Ins.extend(X_train)
    Outs.extend(y_train)

    with open(fname2, 'rb') as inp:
        X_train = pickle.load(inp)
        y_train = pickle.load(inp)
    Ins.extend(X_train)
    Outs.extend(y_train)

    docs = get_input(Ins,Outs)
    X = [extract_features(a_doc) for a_doc in docs[:]]
    y = [get_labels(a_doc) for a_doc in docs[:]]
    print "train" , len(X), len(y)

    with open(path, 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)

def test() :
    file_dict= { 'train' : '../../../data/umass_train_data.pkl', 'test' : '../../../data/umass_test_data.pkl' , 'dev' :'../../../data/umass_val_data.pkl'}
    sync_file_dict =  { 'train' : '../../../data/syn_train.pkl', 'test' : '../../../data/syn_test.pkl' , 'dev' :'../../../data/syn_val.pkl'}

    paths = {'train': '../../../data/feats_train.pkl', 'test' :'../../../data/feats_test.pkl' , 'dev':'../../../data/feats_test.pkl'}

    save_data_pickle(file_dict['train'], sync_file_dict['train'], paths['train'])
    save_data_pickle(file_dict['dev'], sync_file_dict['dev'], paths['dev'])
    save_data_pickle(file_dict['test'], sync_file_dict['test'], paths['test'])

#test()
crf()
