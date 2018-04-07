import tensorflow as tf
import numpy as np
from config import *
from umass_parser import *
from features_tokens import *

def read_dataset(data_type):
    Data = get_data(data_type)

    ##get train data
    strIndex = 0
##    data_feature = np.array([])
##    data_feature = np.expand_dims(data_feature,axis = 0)
##    data_feature = np.expand_dims(data_feature,axis = 0)
##    data_target = np.array([])
##    data_target = np.expand_dims(data_target,axis = 0)
##    data_target = np.expand_dims(data_target,axis = 0)
    for s in Data:
        tokensStr = Data[s][0]
        labelsStr = Data[s][1]
        l = len(tokensStr)
##        tempFeature = np.array([])
##        tempLabel = np.array([])
        for i in range(l):
            #transfer feature output to vector
            wordFeature = Features(tokensStr[i])
            if len(tokensStr[i])<1:
                f = [0.0]*len(config_params['feature_names'])
            else:
                f = wordFeature.vectorize()
            f = np.expand_dims(np.array(f),axis = 0)
            if i==0:
                tempFeature = f
            else:
                tempFeature = np.append(tempFeature,f,axis =0)
            #transfer labels to vector
            oneHot = [0.0]*(len(labels)+1)
            if labelsStr[i] in labels:
                oneHot[labels[labelsStr[i]]] = 1.0
            else:
                oneHot[len(labels)] = 1.0
            oneHot = np.expand_dims(np.array(oneHot),axis = 0)
            if i==0:
                tempLabel = oneHot
            else:
                tempLabel= np.append(tempLabel,oneHot,axis =0)
        #compensate the vector when the string is shorter than mex length
        f = [0.0]*len(config_params['feature_names'])
        f = np.expand_dims(np.array(f),axis = 0)
        oneHot = [0.0]*(len(labels)+1)
        oneHot[len(labels)] = 1.0
        oneHot = np.expand_dims(np.array(oneHot),axis = 0)
        while len(tempFeature)<config_params["max_stream_length"]:
            tempFeature = np.append(tempFeature,f,axis =0)
            tempLabel= np.append(tempLabel,oneHot,axis =0)
        #append the current string result to dataset
        tempFeature = np.expand_dims(tempFeature,axis = 0)
        tempLabel = np.expand_dims(tempLabel,axis = 0)
        if strIndex==0:
            data_feature = tempFeature
            data_target = tempLabel
        else:
            if data_feature.shape[1]<tempFeature.shape[1]:
                print data_feature.shape, tempFeature.shape
            data_feature = np.append(data_feature,tempFeature,axis =0)
            data_target= np.append(data_target,tempLabel,axis =0)
        strIndex = strIndex+1

    return data_feature,data_target

##    ##get validation data
##    valid_data_feature = []
##    valid_data_target = []
##    for s in validData:
##        tokensStr = validData[s][0]
##        labelsStr = validData[s][1]
##        l = len(tokensStr)
##        tempFeature = []
##        tempLabel = []
##        for i in range(l):
##            #transfer feature output to vector
##            wordFeature = Features(tokensStr[i])
##            if len(tokensStr[i])<1:
##                f = [0]*len(config_params['feature_names'])
##            else:
##                f = wordFeature.vectorize()
##            tempFeature.append(f)
##            #transfer labels to vector
##            oneHot = [0]*(len(labels)+1)
##            if labelsStr[i] in labels:
##                oneHot[labels[labelsStr[i]]] = 1
##            else:
##                oneHot[len(labels)] = 1
##            tempLabel.append(oneHot)
##        #compensate the vector when thestring is shorter than mex length
##        f = [0]*len(config_params['feature_names'])
##        oneHot = [0]*(len(labels)+1)
##        oneHot[len(labels)] = 1
##        while len(tempFeature)<config_params["max_stream_length"]:
##            tempFeature.append(f)
##            tempLabel.append(oneHot)
##        #append the current string result to dataset
##        valid_data_feature.append(tempFeature)
##        valid_data_target.append(tempLabel)
##
##    ##get test data
##    test_data_feature = []
##    test_data_target = []
##    for s in testData:
##        tokensStr = testData[s][0]
##        labelsStr = testData[s][1]
##        l = len(tokensStr)
##        tempFeature = []
##        tempLabel = []
##        for i in range(l):
##            #transfer feature output to vector
##            wordFeature = Features(tokensStr[i])
##            if len(tokensStr[i])<1:
##                f = [0]*len(config_params['feature_names'])
##            else:
##                f = wordFeature.vectorize()
##            tempFeature.append(f)
##            #transfer labels to vector
##            oneHot = [0]*(len(labels)+1)
##            if labelsStr[i] in labels:
##                oneHot[labels[labelsStr[i]]] = 1
##            else:
##                oneHot[len(labels)] = 1
##            tempLabel.append(oneHot)
##        #compensate the vector when thestring is shorter than mex length
##        f = [0]*len(config_params['feature_names'])
##        oneHot = [0]*(len(labels)+1)
##        oneHot[len(labels)] = 1
##        while len(tempFeature)<config_params["max_stream_length"]:
##            tempFeature.append(f)
##            tempLabel.append(oneHot)
##        #append the current string result to dataset
##        test_data_feature.append(tempFeature)
##        test_data_target.append(tempLabel)
##            
##        
##        return train_data_feature,train_data_target,valid_data_feature,valid_data_target,test_data_feature,test_data_target
