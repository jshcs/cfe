import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.contrib import rnn
import pickle
from config import *
from umass_parser import *
from features import *
from readDataset import *

#loading data
with open('../../data/train_exp.pkl', 'rb') as inp:
    X_train = pickle.load(inp)
    y_train = pickle.load(inp)

with open('../../data/val_exp.pkl', 'rb') as inp:
    X_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

with open('../../data/test_exp.pickle', 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)

#data_train = BatchGenerator(X_train, y_train, shuffle=False)
# data_valid = BatchGenerator(X_valid, y_valid, shuffle=False)
# data_test = BatchGenerator(X_test, y_test, shuffle=False)

#print data_train


lrate = config_params["lrate"]
num_units = config_params["num_units"]
length = config_params["max_stream_length"]
num_features = len(config_params["feature_names"])
num_classes = len(labels)+1
epochs = config_params["epochs"]
tr_batch_size = config_params["batch_size"]
layer_num = config_params["num_layer"]
decay_rate = config_params["lrate_decay"]
max_grad_norm = 5.0
target_names = ALL_TAGS
target_names.append('unknown')


lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32,[])
wo = tf.Variable(tf.truncated_normal([num_units,num_classes]))
bo = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model_save_path = 'ckpt/lstm.ckpt'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
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
    print "data shape",data.get_shape,"target shape",target.get_shape

#lstm netowk to get the output
lstm_output = lstm(data)

#output of lstm network after last softmax layer
with tf.variable_scope('outputs'):
    label_preds = tf.matmul(lstm_output, wo) + bo
##	print lstm_output.get_shape,wo.get_shape
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


train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())

#test on other data set (valid or test)
def test_epoch(data_x,data_y):
    fetches = [accuracy, cost, label_pred]
    data_size = data_y.shape[0]
    X_batch, y_batch = data_x,data_y
    feed_dict = {data:data_x, target:data_y, batch_size:data_size, keep_prob:1.0}
    _accs, _costs, _pred = sess.run(fetches, feed_dict)
    #F1 result
    _pred = np.argmax(_pred, axis = 2)
    pred = np.reshape(_pred,(1,-1))
    pred = np.squeeze(pred)
    ground_truth = np.argmax(data_y, axis = 2)
    ground_truth = np.reshape(ground_truth,(1,-1))
    ground_truth = np.squeeze(ground_truth)
    _scores = f1_score(ground_truth,pred, average='weighted')
    print('classification report')
    print(classification_report(ground_truth,pred,target_names=target_names))

    return _accs, _costs, _scores


def get_random_batch(data_size,batch_size):
    return np.random.choice(data_size,batch_size)

#begin to train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tr_batch_size = 20
display_num = 10
decay_num = 15
tr_batch_num = int(y_train.shape[0] / tr_batch_size)
saver = tf.train.Saver(max_to_keep=10)
for epoch in xrange(epochs):
    _costs = 0.0
    _accs = 0.0
    _lr = lrate*(decay_rate**(epoch/decay_num))
    for batch in xrange(tr_batch_num):
        fetches = [accuracy, cost, train_op]
        X_batch = X_train[batch*tr_batch_size:(batch+1)*tr_batch_size,:,:]
        y_batch = y_train[batch*tr_batch_size:(batch+1)*tr_batch_size,:,:]
        feed_dict = {data:X_batch, target:y_batch, batch_size:tr_batch_size,lr:_lr, keep_prob:1}
        _acc, _cost, _ = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % display_num == 0:
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print 'the save path is ', save_path
    print 'epoch',epoch+1
    print 'training %d, acc=%g, cost=%g ' % (y_train.shape[0], mean_acc, mean_cost)
    print '**VAL RESULT:'
    val_acc, val_cost,val_score = test_epoch(X_valid,y_valid)
    print '**VAL %d, acc=%g, cost=%g, F1 score = %g' % (y_valid.shape[0], val_acc, val_cost,val_score)

# testing
print '**TEST RESULT:'
test_acc, test_cost,test_score = test_epoch(X_test,y_test)
print '**TEST %d, acc=%g, cost=%g, F1 score = %g' % (y_test.shape[0], test_acc, test_cost, test_score)


print '**DEV RESULT:'
val_acc, val_cost, val_score= test_epoch(X_valid,y_valid)
print '**Test %d, acc=%g, cost=%g, F1 score=%g' % (y_valid.shape[0], val_acc, val_cost, val_score)

