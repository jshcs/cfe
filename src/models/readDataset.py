import tensorflow as tf
import numpy as np
from config import *
from umass_parser import *
from features_tokens import *
import pickle

with open(VOCAB_JNAMES,'rb') as v:
    all_vocab=pickle.load(v)
with open(BIO_SRT,'rb') as v:
    all_bio_vocab=pickle.load(v)

# def read_dataset(data_type):
#     Data = get_data(data_type)
#
#     ##get train data
#     strIndex = 0
#     for s in Data:
#         tokensStr = Data[s][0]
#         labelsStr = Data[s][1]
#         l = len(tokensStr)
#         for i in range(l):
#             #transfer feature output to vector
#             wordFeature = Features(tokensStr[i])
#             if len(tokensStr[i])<1:
#                 f = [0.0]*len(config_params['feature_names'])
#             else:
#                 f = wordFeature.vectorize()
#             f = np.expand_dims(np.array(f),axis = 0)
#             if i==0:
#                 tempFeature = f
#             else:
#                 tempFeature = np.append(tempFeature,f,axis =0)
#             #transfer labels to vector
#             oneHot = [0.0]*(len(labels)+1)
#             if labelsStr[i] in labels:
#                 oneHot[labels[labelsStr[i]]] = 1.0
#             else:
#                 oneHot[len(labels)] = 1.0
#             oneHot = np.expand_dims(np.array(oneHot),axis = 0)
#             if i==0:
#                 tempLabel = oneHot
#             else:
#                 tempLabel= np.append(tempLabel,oneHot,axis =0)
#         #compensate the vector when the string is shorter than mex length
#         f = [0.0]*len(config_params['feature_names'])
#         f = np.expand_dims(np.array(f),axis = 0)
#         oneHot = [0.0]*(len(labels)+1)
#         oneHot[len(labels)] = 1.0
#         oneHot = np.expand_dims(np.array(oneHot),axis = 0)
#         while len(tempFeature)<config_params["max_stream_length"]:
#             tempFeature = np.append(tempFeature,f,axis =0)
#             tempLabel= np.append(tempLabel,oneHot,axis =0)
#         #append the current string result to dataset
#         tempFeature = np.expand_dims(tempFeature,axis = 0)
#         tempLabel = np.expand_dims(tempLabel,axis = 0)
#         if strIndex==0:
#             data_feature = tempFeature
#             data_target = tempLabel
#         else:
#             if data_feature.shape[1]<tempFeature.shape[1]:
#                 print data_feature.shape, tempFeature.shape
#             data_feature = np.append(data_feature,tempFeature,axis =0)
#             data_target= np.append(data_target,tempLabel,axis =0)
#         strIndex = strIndex+1
#
#     return data_feature,data_target



def read_dataset(data_type):
    Data = get_data(data_type)

    ##get train data
    strIndex = 0
    #data_feature=np.zeros((1,config_params["max_stream_length"],len(config_params['feature_names'])))
    #data_target=np.zeros((1,config_params["max_stream_length"],len(labels)+1))
    data_feature=[]
    data_target=[]
    for s in Data:
        tokensStr = Data[s][0]
        labelsStr = np.array(Data[s][1])
        #print tokensStr,labelsStr
        if len(tokensStr)<config_params["max_stream_length"]:
            diff=config_params["max_stream_length"]-len(tokensStr)
            tokensStr+=['<UNK>']*(diff)
            #print config_params["max_stream_length"]-len(tokensStr)
            unk_labels=np.full((diff,),len(labels))
            #print unk_labels,unk_labels.shape
            labelsStr=np.hstack((labelsStr,unk_labels))
            #print labelsStr,labelsStr.shape
        elif len(tokensStr)>config_params["max_stream_length"]:
            tokensStr=tokensStr[:config_params["max_stream_length"]]
            labelsStr=labelsStr[:config_params["max_stream_length"]]
        features_sentence=Features(tokensStr,all_vocab,all_bio_vocab)
        vectorized_features=features_sentence.get_features()
        #vectorized_features=np.expand_dims(vectorized_features,axis=0)
        onehot_labels=np.eye(len(labels)+1)[labelsStr]

        #onehot_labels=np.expand_dims(onehot_labels,axis=0)
        #print vectorized_features.shape,onehot_labels.shape,data_feature.shape,data_target.shape
        #data_feature=np.concatenate((data_feature,vectorized_features),axis=0)
        #data_target=np.concatenate((data_target,onehot_labels),axis=0)
        data_feature.append(vectorized_features)
        data_target.append(onehot_labels)

        #print type(data_feature),type(data_target)
    return np.array(data_feature),np.array(data_target)

# r=read_dataset("train")
# print r[0],r[1]

