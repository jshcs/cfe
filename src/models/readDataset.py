import tensorflow as tf
import numpy as np
from config import *
from umass_parser import *
from features_tokens import *
import pickle
import time
from gensim.models.keyedvectors import KeyedVectors
#import simstring

# with open(VOCAB_JNAMES,'rb') as v:
#     all_vocab=pickle.load(v)
with open(BIO_SRT,'rb') as v:
    all_bio_vocab=pickle.load(v)
WV=KeyedVectors.load_word2vec_format(WE_BIN, binary=True)
sorted_fname=read_sorted_file_into_array(SORTED_FPERSON_FNAME)
sorted_lname=read_sorted_file_into_array(SORTED_LPERSON_FNAME)
bio_dict={voc:1 for voc in all_bio_vocab}
# journal_dict={voc:1 for voc in all_vocab}

#sorted_journals=read_sorted_file_into_array(COMBINED_JNAMES)
sorted_journals_db=simstring.reader(DB_JNAMES)
sorted_journals_db.measure=SS_METRIC
sorted_journals_db.threshold=SS_THRESHOLD
# sorted_journals=[[t.lower() for t in ele.split()] for ele in sorted_journals]
#print sorted_journals

def read_dataset(data_type):
    Data = get_data(data_type)
    c=0
    max_length=config_params["max_stream_length"]
    n_features=len(config_params["feature_names"])+EMD_SIZE-1
    #print 'n_features',n_features

    data_feature=np.zeros((1,max_length,n_features))
    print data_feature.shape
    data_target=[]
    total_time=0
    for s in Data:
        start=time.time()
        c+=1
        tokensStr = Data[s][0]
        labelsStr = Data[s][1]
        len_string=len(tokensStr)

        if len(tokensStr)>=config_params["max_stream_length"]:
            tokensStr=tokensStr[:config_params["max_stream_length"]]
            labelsStr=labelsStr[:config_params["max_stream_length"]]
        #print tokensStr
        features_sentence=Features(tokensStr,sorted_fname,sorted_lname,bio_dict,sorted_journals_db,WV)
        # print s
        # print

        vectorized_features=features_sentence.get_features()
        #print len(vectorized_features),len(vectorized_features[0])
        vectorized_features=np.array(vectorized_features)
        #print vectorized_features.shape
        if len(tokensStr)<config_params["max_stream_length"]:
            diff=config_params["max_stream_length"]-len(tokensStr)
            extra_features = np.zeros((diff,n_features))
            vectorized_features = np.concatenate((vectorized_features, extra_features), axis=0)
            labelsStr+=[len(labels)]*diff
        vectorized_features = np.expand_dims(vectorized_features, axis=0)
        # print vectorized_features
        data_feature=np.concatenate((data_feature,vectorized_features),axis=0)

        # print vectorized_features.
        #print len(vectorized_features),len(vectorized_features[0]),type(vectorized_features)
        labelsStr=np.array(labelsStr)
        onehot_labels=np.eye(len(labels)+1)[labelsStr]
        #data_feature.append(vectorized_features)
        #print 'data_feature.shape',data_feature.shape
        data_target.append(onehot_labels)
        end_time=time.time()
        total_time+=(end_time-start)
        print "Sentences done:",c,"in:",(end_time-start),"total time:",total_time,"avg time:",(float(total_time)/c),"length:",len_string
    return data_feature[1:,:,:],np.array(data_target)

def read_bibtex_dataset(Data):
    c=0
    max_length=config_params["max_stream_length"]
    n_features=len(config_params["feature_names"])+EMD_SIZE-1
    #print 'n_features',n_features

    data_feature=np.zeros((1,max_length,n_features))
    print data_feature.shape
    data_target=[]
    total_time=0
    for s in Data:
        start=time.time()
        c+=1
        tokensStr = Data[s][0]
        labelsStr = Data[s][1]
        len_string=len(tokensStr)

        if len(tokensStr)>=config_params["max_stream_length"]:
            tokensStr=tokensStr[:config_params["max_stream_length"]]
            labelsStr=labelsStr[:config_params["max_stream_length"]]
        features_sentence=Features(tokensStr,sorted_fname,sorted_lname,bio_dict,sorted_journals_db,WV)

        vectorized_features=features_sentence.get_features()

        vectorized_features=np.array(vectorized_features)

        if len(tokensStr)<config_params["max_stream_length"]:
            diff=config_params["max_stream_length"]-len(tokensStr)
            extra_features = np.zeros((diff,n_features))
            vectorized_features = np.concatenate((vectorized_features, extra_features), axis=0)
            labelsStr+=[len(labels)]*diff
        vectorized_features = np.expand_dims(vectorized_features, axis=0)
        data_feature=np.concatenate((data_feature,vectorized_features),axis=0)

        labelsStr=np.array(labelsStr)
        onehot_labels=np.eye(len(labels)+1)[labelsStr]

        data_target.append(onehot_labels)
        end_time=time.time()
        total_time+=(end_time-start)
        print "Sentences done:",c,"in:",(end_time-start),"total time:",total_time,"avg time:",(float(total_time)/c),"length:",len_string
    return data_feature[1:,:,:],np.array(data_target)

# s=time.time()
# f,t=read_dataset("test")
# print type(f),f.shape
# e=time.time()
#
# print r[0],r[1]
# print "Time:",(e-s)
