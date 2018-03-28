import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pickle
from config import *
from umass_parser import *
from features import *
from readDataset import *
from BatchGenerator import *

#loading data
with open('../../data/train.pkl', 'rb') as inp:
    X_train = pickle.load(inp)
    y_train = pickle.load(inp)

with open('../../data/val.pkl', 'rb') as inp:
    X_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

with open('../../data/test.pkl', 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)

data_train = BatchGenerator(X_train, y_train, shuffle=False)
data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_test, y_test, shuffle=False)

lrate = config_params["lrate"]
num_units = config_params["num_units"]
length = config_params["max_stream_length"]
num_features = len(config_params["feature_names"])
num_classes = len(labels)+1
epochs = config_params["epochs"]
tr_batch_size = config_params["batch_size"]
layer_num = config_params["num_layer"]
max_grad_norm = 5.0

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32,[])
wo = tf.Variable(tf.truncated_normal([num_units,num_classes]))
bo = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model_save_path = 'ckpt/bi-lstm.ckpt'

#lstm cell
def lstm_cell():
    cell = rnn.LSTMCell(num_units, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

#lstm network
def lstm(tokens):
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell_fw,tokens, dtype=tf.float32)
    output = tf.reshape(outputs, [-1,num_units])
    return output

#input placeholder
with tf.variable_scope('Inputs'):
    data = tf.placeholder(tf.float32, shape=(None, length, num_features))
    target = tf.placeholder(tf.float32, shape=(None, length, num_classes))

#lstm netowk to get the output
lstm_output = lstm(data)

#output of lstm network after last softmax layer
with tf.variable_scope('outputs'):
    label_preds = tf.matmul(lstm_output, wo) + bo
    print lstm_output.get_shape,wo.get_shape
    label_pred = tf.reshape(label_preds, [-1,length, num_classes])

#check predict result against ground truth
correct_prediction = tf.equal(tf.argmax(label_pred, 2),tf.argmax(target,2))
#get accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = target, logits = label_pred))

#grandient desent
tvars = tf.trainable_variables() 
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)


train_op = optimizer.apply_gradients( zip(grads, tvars),
    global_step=tf.contrib.framework.get_or_create_global_step())

#test on other data set (valid or test)
def test_epoch(dataset):
    _batch_size = 50
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in xrange(batch_num):
        X_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs:X_batch, y_inputs:y_batch, batch_size:_batch_size, keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost    
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

#begin to train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tr_batch_size = 100 
display_num = 10 
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  
display_batch = int(tr_batch_num / display_num) 
saver = tf.train.Saver(max_to_keep=10)
for epoch in xrange(epochs):
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    for batch in xrange(tr_batch_num): 
        fetches = [accuracy, cost, train_op]
        X_batch, y_batch = data_train.next_batch(tr_batch_size)
        print X_batch.shape,y_batch.shape
        feed_dict = {data:X_batch, target:y_batch, batch_size:tr_batch_size,lr:lrate, keep_prob:0.5}
        _acc, _cost, _ = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print '\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost)
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num 
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 10 == 0:
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'the save path is ', save_path
    print '\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost)
    print 'Epoch training %d, acc=%g, cost=%g' % (data_train.y.shape[0], mean_acc, mean_cost)        
# testing
print '**TEST RESULT:'
test_acc, test_cost = test_epoch(data_valid)
print '**Test %d, acc=%g, cost=%g' % (data_valid.y.shape[0], test_acc, test_cost)
