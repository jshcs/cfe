import tensorflow as tf
from bibtex2Dict import *
import numpy as np
from config import *
from umass_parser import *
from features import *
from readDataset import *
import pickle
import os

def main():
    #initial parameter
    length = config_params["max_stream_length"]
    num_features = len(config_params["feature_names"])
    num_classes = len(labels)+1
    epochs = config_params["epochs"]
    batch_size = config_params["batch_size"]

    # reading umass data
    X_train,y_train = read_dataset("train")
    X_valid,y_valid = read_dataset("dev")
    X_test,y_test= read_dataset("test")

    # print 'writing'
    # print 'train'
    # print train_token.shape,train_label.shape
    # with open('../../data/we_pickles/umass_train.pkl', 'wb') as outp:
    #     pickle.dump(train_token, outp)
    #     pickle.dump(train_label, outp)
    #
    # print 'val'
    # print val_token.shape,val_label.shape
    # with open('../../data/we_pickles/umass_val.pkl', 'wb') as outp:
    #     pickle.dump(val_token, outp)
    #     pickle.dump(val_label, outp)
    #
    # print 'test'
    # print test_token.shape,test_label.shape
    # with open('../../data/we_pickles/umass_test.pickle', 'wb') as outp:
    #     pickle.dump(np.array(test_token), outp)
    #     pickle.dump(np.array(test_label), outp)

    # reading bibtex data

    for style in styleFile:
        trainDict,valDict,testDict = dict_of_style(style)
        bibtex_X_train,bibtex_y_train = read_bibtex_dataset(trainDict)
        bibtex_X_valid,bibtex_y_valid = read_bibtex_dataset(valDict)
        bibtex_X_test,bibtex_y_test = read_bibtex_dataset(testDict)

        # print 'writing'
        # print 'train'
        # print train_token.shape,train_label.shape
        # with open('../../data/we_pickles/'+style+'_train.pkl', 'wb') as outp:
        #     pickle.dump(train_token, outp)
        #     pickle.dump(train_label, outp)
        #
        # print 'val'
        # print val_token.shape,val_label.shape
        # with open('../../data/we_pickles/'+style+'_val.pkl', 'wb') as outp:
        #     pickle.dump(val_token, outp)
        #     pickle.dump(val_label, outp)
        #
        # print 'test'
        # print test_token.shape,test_label.shape
        # with open('../../data/we_pickles/'+style+'_test.pickle', 'wb') as outp:
        #     pickle.dump(np.array(test_token), outp)
        #     pickle.dump(np.array(test_label), outp)

        X_train=np.concatenate((X_train,bibtex_X_train),axis=0)
        y_train=np.concatenate((y_train,bibtex_y_train),axis=0)
        X_valid=np.concatenate((X_valid,bibtex_X_valid),axis=0)
        y_valid=np.concatenate((y_valid,bibtex_y_valid),axis=0)
        X_test=np.concatenate((X_test,bibtex_X_test),axis=0)
        y_test=np.concatenate((y_test,bibtex_y_test),axis=0)

    np.save('../../data/we_npy/combined_X_train.npy',X_train,allow_pickle=False)
    np.save('../../data/we_npy/combined_y_train.npy',y_train,allow_pickle=False)
    np.save('../../data/we_npy/combined_X_valid.npy',X_valid,allow_pickle=False)
    np.save('../../data/we_npy/combined_y_valid.npy',y_valid,allow_pickle=False)
    np.save('../../data/we_npy/combined_X_test.npy',X_test,allow_pickle=False)
    np.save('../../data/we_npy/combined_y_test.npy',y_test,allow_pickle=False)

    print X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape,y_test.shape

if __name__ == '__main__':
    main()
