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

	print 'writing'
	print 'train'
	print train_token.shape,train_label.shape
	with open('../../data/train.pkl', 'wb') as outp:
		pickle.dump(train_token, outp)
		pickle.dump(train_label, outp)

	print 'val'
	print val_token.shape,val_label.shape
	with open('../../data/val.pkl', 'wb') as outp:
		pickle.dump(val_token, outp)
		pickle.dump(val_label, outp)

	print 'test'
	print np.array(val_token).shape,np.array(val_label).shape
	with open('../../data/test.pickle', 'wb') as outp:
		pickle.dump(np.array(test_token), outp)
		pickle.dump(np.array(test_label), outp)

	# print 'reading'
	#
	# with open('../../data/train.pkl', 'rb') as inp:
	# X_train = pickle.load(inp)
	# y_train = pickle.load(inp)
	# print 'train'
	# print X_train.shape,y_train.shape
	#
	# with open('../../data/val.pkl', 'rb') as inp:
	# 	X_valid = pickle.load(inp)
	# y_valid = pickle.load(inp)
	# print 'val'
	# print X_valid.shape,y_valid.shape

##    with open('../../data/test.pickle', 'rb') as inp:
##        X_test = pickle.load(inp)
##	y_test = pickle.load(inp)
##
##    print 'test'	
##    print X_test.shape,y_test.shape

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
