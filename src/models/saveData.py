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
    
    #reading data
    train_token,train_label = read_dataset("train")
    val_token,val_label = read_dataset("dev")
    test_token,test_label = read_dataset("test")

    with open('../../data/train.pkl', 'wb') as outp:
        pickle.dump(np.asarray(train_token), outp)
        pickle.dump(np.asarray(train_label), outp)

    with open('../../data/val.pkl', 'wb') as outp:
        pickle.dump(np.asarray(val_token), outp)
        pickle.dump(np.asarray(val_label), outp)

    with open('../../data/test.pkl', 'wb') as outp:
        pickle.dump(np.asarray(test_token), outp)
        pickle.dump(np.asarray(test_label), outp)


    
if __name__ == '__main__':
    main()
