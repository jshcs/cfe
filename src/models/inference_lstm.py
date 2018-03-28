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

    #initial tf placeholder
##    data = tf.placeholder(tf.float32, shape=(None, length, num_features))
##    target = tf.placeholder(tf.float32, shape=(None, length, num_classes))
##
##    #lstm network
##    model = LSTM(data,target)
##    init = tf.global_variables_initializer()
##    sess = tf.Session()
##    sess.run(init)
##
##    batch = len(train_token)//batch_size
##    if len(train_token)%batch_size>0:
##        batch = batch+1
##    for epoch in range(epochs):
##        print(epoch)
##        for b in range(batch):
##            if b==batch-1:
##                token_batch = train_token[b*batch_size:]
##                label_batch = train_label[b*batch_size:]
##            else:
##                token_batch = train_token[b*batch_size:(b+1)*batch_size]
##                label_batch = train_label[b*batch_size:(b+1)*batch_size]
##            sess.run(model.opt,
##                     {data: token_batch, target: label_batch})
##        if (epoch+1)%10==0:
##            error = sess.run(model.error,
##                             {data: train_token, target: train_label})
##        print('Epoch {:2d} error on valid data {:3.1f}%'.format(epoch + 1, 100 * error))
##
##    valError = sess.run(model.error,
##                        {data:val_token, target:val_label})
##    print('Validation error on valid data {:3.1f}%'.format(100 * error))


    
if __name__ == '__main__':
    main()
