import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.contrib import rnn
import pickle
from config import *
#from umass_parser import *
#from readDataset import *
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import itertools

print "starting to gather data...."

X_train=np.load('../../data/we_npy/combined_X_train.npy')
y_train=np.load('../../data/we_npy/combined_y_train.npy')
X_valid=np.load('../../data/we_npy/combined_X_valid.npy')
y_valid=np.load('../../data/we_npy/combined_y_valid.npy')
X_test=np.load('../../data/we_npy/combined_X_test.npy')
y_test=np.load('../../data/we_npy/combined_y_test.npy')

print X_train.shape,X_valid.shape,X_test.shape,y_train.shape,y_valid.shape,y_test.shape


lrate = config_params["lrate"]
num_units = config_params["num_units"]
length = config_params["max_stream_length"]
num_features = len(config_params["feature_names"])+EMD_SIZE-1
num_classes = len(labels)+1
epochs = config_params["epochs"]
tr_batch_size = config_params["batch_size"]
layer_num = config_params["num_layer"]
decay_rate = config_params["lrate_decay"]
max_grad_norm = 5.0
target_names = ALL_TAGS
target_names.append('unknown')


with tf.device("/cpu:0"):
    lr = tf.placeholder(tf.float32, []) 
    keep_prob = tf.placeholder(tf.float32, [])
    batch_size = tf.placeholder(tf.int32,[])
    wo = tf.Variable(tf.truncated_normal([num_units,num_classes]))
    bo = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    model_save_path='ckpt/lstm'

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
with tf.device("/cpu:0"):
    with tf.variable_scope('Inputs'):
        data = tf.placeholder(tf.float32, shape=(None, length, num_features))
        target = tf.placeholder(tf.float32, shape=(None, length, num_classes))
        print "data shape",data.get_shape,"target shape",target.get_shape

#lstm netowk to get the output
lstm_output = lstm(data)

#output of lstm network after last softmax layer
with tf.device("/cpu:0"):
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
train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.train.get_or_create_global_step())
#test on other data set (valid or test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='LSTM Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test_epoch(data_x,data_y,final):
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
    print "pred:",pred,"ground:",ground_truth
    _scores = f1_score(ground_truth,pred, average='weighted')
    print('classification report')
    print(classification_report(ground_truth,pred,target_names=target_names))
    # print(classification_report(
    #     ground_truth,pred,labels=ALL_TAGS,digits=2
    # ))

    #labels=['title','volume','year','journal','person','pages']
    if final==True:
        #np.set_printoptions(precision=2)
        cm=confusion_matrix(ground_truth,pred)
        print 'cm shape',cm.shape
        # print 'cm shape',cm.shape
        cm=cm[:len(labels),:len(labels)]
        print 'cm shape',cm.shape

        plt.figure()
        plot_confusion_matrix(cm,classes=target_names[:-1],
                              title='LSTM consusion matrix w/o normalization')
        plt.savefig('lstm_cm_no_norm.png')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm,classes=target_names[:-1],normalize=True,
                              title='LSTM confusion matrix')

        plt.savefig('lstm_cm_norm.png')

    return _accs, _costs, _scores


def get_random_batch(data_size,batch_size):
    return np.random.choice(data_size,batch_size)

#begin to train

print "*"*30
print "Starting training...."
#
# #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# #sess.run(tf.global_variables_initializer())
# tr_batch_size = 100
display_num = 10
decay_num = 20
tr_batch_num = int(y_train.shape[0] / tr_batch_size)
saver = tf.train.Saver(max_to_keep=10)
all_results=[]
max_f1=0
bestModel=0
for lrate in LR_RANGE:
    for decay_rate in DECAY_RATE:
        sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
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
            # if (epoch + 1) % display_num == 0:
#             #     save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
#             #     print 'the save path is ', save_path
#             # print 'epoch',epoch+1
#             # print 'training %d, acc=%g, cost=%g ' % (y_train.shape[0], mean_acc, mean_cost)
#             # print '**VAL RESULT:'
#             # val_acc, val_cost,val_score = test_epoch(X_valid,y_valid)
#             # print '**VAL %d, acc=%g, cost=%g, F1 score = %g' % (y_valid.shape[0], val_acc, val_cost,val_score)
#
            if (epoch+1)%display_num==0:
                print 'learning rate:',lrate,'decay_rate:',decay_rate
                print 'epoch',epoch+1
                print 'training %d, acc=%g, cost=%g '%(y_train.shape[0],mean_acc,mean_cost)
            if (epoch+1)>=50:
                print '**VAL RESULT:'
                val_acc,val_cost,val_score=test_epoch(X_valid,y_valid,False)
                print '**VAL %d, acc=%g, cost=%g, F1 score = %g'%(y_valid.shape[0],val_acc,val_cost,val_score)
                all_results.append({'lr':lrate,'decay_rate':decay_rate,'epoch':epoch+1,'valAcc':val_acc,'valScore':val_score})
                if max_f1<val_score:
                    max_f1=val_score
                    bestModel=len(all_results)
                #save model
                save_path=saver.save(sess,model_save_path+'-lr_%g-dr_%g_ep%d.ckpt'%(lrate,decay_rate,epoch+1))

# # # testing
# # print '**TEST RESULT:'
# # test_acc, test_cost,test_score = test_epoch(X_test,y_test)
# # print '**TEST %d, acc=%g, cost=%g, F1 score = %g' % (y_test.shape[0], test_acc, test_cost, test_score)
#
#
#
for vRes in all_results:
    print vRes

#check best model and apply on test model
##tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ##    #get best model
    l=all_results[bestModel-1]['lr']
    dr=all_results[bestModel-1]['decay_rate']
    i=all_results[bestModel-1]['epoch']
    # l = 0.001
    # dr = 0.85
    # i = 50
    print 'lrate:',l
    print 'decay rate',dr
    print 'epochs',i
    saver.restore(sess,model_save_path+'-lr_%g-dr_%g_ep%d.ckpt'%(l,dr,i))
    #evaluation the model on test set
    test_acc,test_cost,test_score=test_epoch(X_test,y_test,True)
    print '**TEST RESULT:'
    print '**TEST %d, acc=%g, cost=%g, F1 score = %g'%(y_test.shape[0],test_acc,test_cost,test_score)


