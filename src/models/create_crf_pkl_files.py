import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np
from config import *
import pickle

from def_features import *

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

def create_feats_pkl(paths_dict , key2):
    keys = paths_dict.keys()
    for key in keys :
        Ins = []
        Outs = []
        with open(paths_dict[key],"rb") as inp:
            X_t = pickle.load(inp)
            y_t = pickle.load(inp)
        Ins.extend(X_t)
        Outs.extend(y_t)
        docs = get_input(Ins , Outs)
        X = [extract_features(a_doc) for a_doc in docs[:]]
        y = [get_labels(a_doc) for a_doc in docs[:]]

        print paths_dict[key] , key , len(X), len(y)

        with open(get_output_fname(key,key2), 'wb') as outp:
            pickle.dump(X, outp)
            pickle.dump(y, outp)

        with open(get_output_fname(key,key2), 'rb') as outp:
            X = pickle.load(outp)
            y = pickle.load(outp)

        print get_output_fname(key,key2) , len(X), len(y)


create_feats_pkl(input_unseen_path,'unseen')
#create_feats_pkl(input_path_umass,'umass')
#create_feats_pkl(input_path_comb,'comb')

