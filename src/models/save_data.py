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
    train_token,train_label = read_dataset("train")
    val_token,val_label = read_dataset("dev")
    test_token,test_label = read_dataset("test")

    print 'writing'
    print 'train'
    print train_token.shape,train_label.shape
    with open('../../data/umass_train.pkl', 'wb') as outp:
        pickle.dump(train_token, outp)
        pickle.dump(train_label, outp)

    print 'val'
    print val_token.shape,val_label.shape
    with open('../../data/umass_val.pkl', 'wb') as outp:
        pickle.dump(val_token, outp)
        pickle.dump(val_label, outp)

    print 'test'
    print test_token.shape,test_label.shape
    with open('../../data/umass_test.pickle', 'wb') as outp:
        pickle.dump(np.array(test_token), outp)
        pickle.dump(np.array(test_label), outp)
    #
    # # reading bibtex data

    for style in styleFile:
        trainDict,valDict,testDict = dict_of_style(style)
        train_token,train_label = read_bibtex_dataset(trainDict)
        val_token,val_label = read_bibtex_dataset(valDict)
        test_token,test_label = read_bibtex_dataset(testDict)

        print 'writing'
        print 'train'
        print train_token.shape,train_label.shape
        with open('../../data/'+style+'_train.pkl', 'wb') as outp:
            pickle.dump(train_token, outp)
            pickle.dump(train_label, outp)

        print 'val'
        print val_token.shape,val_label.shape
        with open('../../data/'+style+'_val.pkl', 'wb') as outp:
            pickle.dump(val_token, outp)
            pickle.dump(val_label, outp)

        print 'test'
        print test_token.shape,test_label.shape
        with open('../../data/'+style+'_test.pickle', 'wb') as outp:
            pickle.dump(np.array(test_token), outp)
            pickle.dump(np.array(test_label), outp)

if __name__ == '__main__':
    main()
