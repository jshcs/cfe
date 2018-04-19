import tensorflow as tf

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

    # reading data
    train_token,train_label = read_dataset("train")
    val_token,val_label = read_dataset("dev")
    test_token,test_label = read_dataset("test")

    print 'writing'
    print 'train'
    print train_token.shape,train_label.shape
    with open('../../data/train_exp.pkl', 'wb') as outp:
        pickle.dump(train_token, outp)
        pickle.dump(train_label, outp)

    print 'val'
    print val_token.shape,val_label.shape
    with open('../../data/val_exp.pkl', 'wb') as outp:
        pickle.dump(val_token, outp)
        pickle.dump(val_label, outp)

    print 'test'
    print test_token.shape,test_label.shape
    with open('../../data/test_exp.pickle', 'wb') as outp:
        pickle.dump(np.array(test_token), outp)
        pickle.dump(np.array(test_label), outp)

if __name__ == '__main__':
    main()
