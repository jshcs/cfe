import tensorflow as tf
import numpy as np
from config import *
from umass_parser import *
from features_tokens import *
import pickle
import time

with open(VOCAB_JNAMES,'rb') as v:
    all_vocab=pickle.load(v)
with open(BIO_SRT,'rb') as v:
    all_bio_vocab=pickle.load(v)
sorted_fname=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
sorted_lname=read_sorted_file_into_array(SORTED_LPERSON_FNAME)


def read_dataset(data_type):
    Data = get_data(data_type)
    c=0
    data_feature=[]
    data_target=[]
    for s in Data:
        c+=1
        tokensStr = Data[s][0]
        labelsStr = Data[s][1]

        if len(tokensStr)<config_params["max_stream_length"]:
            diff=config_params["max_stream_length"]-len(tokensStr)
            tokensStr+=['<UNK>']*(diff)
            labelsStr+=[len(labels)]*diff

        elif len(tokensStr)>config_params["max_stream_length"]:
            tokensStr=tokensStr[:config_params["max_stream_length"]]
            labelsStr=labelsStr[:config_params["max_stream_length"]]
        print tokensStr
        features_sentence=Features(tokensStr,sorted_fname,sorted_lname,all_vocab,all_bio_vocab)
        vectorized_features=features_sentence.get_features()
        labelsStr=np.array(labelsStr)
        onehot_labels=np.eye(len(labels)+1)[labelsStr]
        data_feature.append(vectorized_features)
        data_target.append(onehot_labels)
        print "Sentences done:",c
    return np.array(data_feature),np.array(data_target)

# s=time.time()
# r=read_dataset("train")
# e=time.time()
#
# print r[0],r[1]
# print "Time:",(e-s)
