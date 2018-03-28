import tensorflow as tf
from config import *
from umass_parser import *
from features import *

def read_dataset(data_type):
    Data = get_data(data_type)

    ##get train data
    data_feature = []
    data_target = []
    for s in Data:
        tokensStr = Data[s][0]
        labelsStr = Data[s][1]
        l = len(tokensStr)
        tempFeature = []
        tempLabel = []
        for i in range(l):
            #transfer feature output to vector
            wordFeature = Features(tokensStr[i])
            if len(tokensStr[i])<1:
                f = [0.0]*len(config_params['feature_names'])
            else:
                f = wordFeature.vectorize()
            tempFeature.append(np.array(f))
            #transfer labels to vector
            oneHot = [0.0]*(len(labels))
            if labelsStr[i] in labels:
                oneHot[labels[labelsStr[i]]] = 1.0
            else:
                oneHot[len(labels)-1] = 1.0
            tempLabel.append(np.array(oneHot))
        #compensate the vector when thestring is shorter than mex length
        f = [0.0]*len(config_params['feature_names'])
        oneHot = [0.0]*(len(labels))
        oneHot[len(labels)-1] = 1.0
        while len(tempFeature)<config_params["max_stream_length"]:
            tempFeature.append(np.array(f))
            tempLabel.append(np.array(oneHot))
        #append the current string result to dataset
        data_feature.append(np.array(tempFeature))
        data_target.append(np.array(tempLabel))

    return np.array(data_feature),np.array(data_target)

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
