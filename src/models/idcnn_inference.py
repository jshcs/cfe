import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from config import *
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import update_results
import json
print 'start running'

#loading data
data_zip=np.load('../../data/we_npy_no_bio/umass_dataset.npz')
X_train=data_zip['X_train.npy']
y_train=data_zip['y_train.npy']
X_valid=data_zip['X_valid.npy']
y_valid=data_zip['y_valid.npy']
X_test=data_zip['X_test.npy']
y_test=data_zip['y_test.npy']
dataset='umass'
test_set='umass'
# test_set='umass_heldout'
# test_set='combined'
# test_set='heldout'
##X_train=np.load('../../data/we_npy/combined_X_train.npy')
##y_train=np.load('../../data/we_npy/combined_y_train.npy')
##X_valid=np.load('../../data/we_npy/combined_X_valid.npy')
##y_valid=np.load('../../data/we_npy/combined_y_valid.npy')
##X_test=np.load('../../data/we_npy/combined_X_test.npy')
##y_test=np.load('../../data/we_npy/combined_y_test.npy')
##dataset='combined'

print 'input size'
print X_train.shape,X_valid.shape,X_test.shape
print 'label size'
print y_train.shape,y_valid.shape,y_test.shape

lrate = config_params["lrate"]
length = config_params["max_stream_length"]
num_features = len(config_params["feature_names"])+EMD_SIZE-1
num_classes = len(labels)+1
epochs = config_params["epochs"]
batch_size = config_params["batch_size"]
layer_num = config_params["num_layer"]
num_filter = config_params["num_units"]
decay_rate = config_params["lrate_decay"]
filter_width = config_params["filter_width"]
repeat_times = config_params["repeat_times"]
max_grad_norm = 5.0
target_names = ALL_TAGS
##target_names.append('unknown')

lr = tf.placeholder(tf.float32, [])
dropout = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32,[])

layers = [{'dilation': 1},{'dilation': 1},{'dilation': 2}]

model_save_path = 'ckpt/idcnn'
##sess = tf.Session()
##sess.run(tf.global_variables_initializer())

#input placeholder
with tf.variable_scope('Inputs'):
    data = tf.placeholder(tf.float32, shape=(None, None, num_features))
    target = tf.placeholder(tf.float32, shape=(None, None, num_classes))

#id-cnn network
def idcnn(tokens):
    inputs = tf.expand_dims(tokens, 1)
    reuse = False

    if dropout == 1.0:
        reuse = True
    with tf.variable_scope("idcnn"):
        with tf.device("/gpu:0"):
            filter_weights = tf.get_variable("idcnn_filter",
                    shape=[1, filter_width, num_features,num_filter],
                    initializer=initializers.xavier_initializer())

            layerInput = tf.nn.conv2d(inputs,filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
        finalOutFromLayers = []
        totalWidthForLastDim = 0
        for j in range(repeat_times):
            for i in range(len(layers)):
                dilation = layers[i]['dilation']
                if i == (len(layers) - 1):
                    isLast = True 
                else:
                    isLast = False
                with tf.device("/gpu:0"):
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        w = tf.get_variable("filterW",
                                            shape=[1, filter_width, num_filter,
                                                   num_filter],
                                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,w,rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                    if isLast:
                        finalOutFromLayers.append(conv)
                        totalWidthForLastDim += num_filter
                    layerInput = conv
        finalOut = tf.concat(axis=3, values=finalOutFromLayers)
        keepProb = 1.0 if reuse else 0.5
        finalOut = tf.nn.dropout(finalOut, keepProb)

        finalOut = tf.squeeze(finalOut, [1])
        finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
        cnn_output_width = totalWidthForLastDim

    with tf.device("/gpu:0"):
        with tf.variable_scope("logits"):
            W = tf.get_variable("W", shape=[cnn_output_width,num_classes],
                                dtype=tf.float32, initializer=initializers.xavier_initializer())
            b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[num_classes]))

            pred = tf.nn.xw_plus_b(finalOut, W, b)

            preds = tf.reshape(pred, [-1, length, num_classes])

    return preds

#id-cnn network output
idcnn_output = idcnn(data)

#check predict result against ground truth
correct_prediction = tf.equal(tf.argmax(idcnn_output, 2),tf.argmax(target,2))
#get accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = target, logits = idcnn_output)) 


#grandient desent
tvars = tf.trainable_variables() 
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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
    
#test on other data set (valid or test)
##test_batch_size = 100
def testModule(data_x,data_y,final,plot_name):
##    test_batch_num = data_x.shape[0]/test_batch_size
##    if data_x.shape[0]%test_batch_size>0:
##        test_batch_num = test_batch_num +1
##    for n in range(test_batch_num):
    fetches = [accuracy, cost, idcnn_output]
    data_size = data_y.shape[0]
    feed_dict = {data:data_x, target:data_y, dropout:1.0}
    _accs, _costs, _pred = sess.run(fetches, feed_dict)
    #save result
    if final==True:
        np.save('final_model/test_result.npy',_pred,allow_pickle=False)
    #F1 result
    _pred = np.argmax(_pred, axis = 2)
    pred = np.reshape(_pred,(1,-1))
    pred = np.squeeze(pred)
    ground_truth = np.argmax(data_y, axis = 2)
    ground_truth = np.reshape(ground_truth,(1,-1))
    ground_truth = np.squeeze(ground_truth)
    _scores = f1_score(ground_truth,pred, average='weighted')
    print('classification report')
    clf_rep=precision_recall_fscore_support(ground_truth,pred)
    out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }
    print(classification_report(ground_truth,pred,target_names=target_names))
    if final==True:
        cm=confusion_matrix(ground_truth,pred)
        print 'cm shape',cm.shape
        cm = cm[:len(labels),:len(labels)]
        print 'cm shape',cm.shape
        plt.figure()
        plot_confusion_matrix(cm,classes=target_names[:-1],
                              title='ID-CNN confusion matrix w/o normalization')
        plt.savefig('idcnn_cm_no_norm'+plot_name+'.png')

        plt.figure()
        plot_confusion_matrix(cm, classes=target_names[:-1], normalize=True,
                              title='ID-CNN confusion matrix')
        plt.savefig('idcnn_cm_norm_'+plot_name+'.png')

    return _accs, _costs, _scores,out_dict

#begin to train
batch_size = 50
display_num = 10
decay_num = 20
tr_batch_num = y_train.shape[0]/batch_size
if y_train.shape[0]%batch_size>0:
    tr_batch_num = tr_batch_num+1
saver = tf.train.Saver(max_to_keep=100)

valResult = []
bestScore = 0.0



for l in LR_RANGE:
    for d in DECAY_RATE:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                total_accs = 0.0
                total_loss = 0.0
                new_index = np.random.permutation(y_train.shape[0])
                trainX = X_train[new_index]
                trainY = y_train[new_index]
                _lr = l*(d**(i/decay_num))
                for b in range(tr_batch_num):
                    #get batch
                    if b==tr_batch_num-1:
                        x_batch = trainX[b*batch_size:,:,:]
                        y_batch = trainY[b*batch_size:,:]
                    else:
                        x_batch = trainX[b*batch_size:(b+1)*batch_size,:,:]
                        y_batch = trainY[b*batch_size:(b+1)*batch_size,:]
                    #update gradient
                    fetches = [accuracy, cost, train_op]
                    feed_dict = {data:x_batch,target:y_batch,lr:_lr,dropout:1}
                    acc, loss, _ = sess.run(fetches, feed_dict)
                    total_accs = total_accs+acc
                    total_loss = total_loss+loss
                mean_acc = total_accs/tr_batch_num
                mean_loss = total_loss/tr_batch_num
                if (i + 1) % display_num == 0:
                    print 'learning rate:',l,'decay_rate:',d
                    print 'epoch',i+1
                    print 'training %d, acc=%g, cost=%g ' % (y_train.shape[0], mean_acc, mean_loss)
                if (i+1)>=epochs:
                    print '**VAL RESULT:'
                    val_acc, val_cost,val_score,out_dict = testModule(X_valid,y_valid,final=False,plot_name=test_set)
                    f1_scores=out_dict['f1-score'][:-1]
                    support=out_dict['support'][:-1]
                    updated_score=sum([f1_scores[i]*support[i] for i in range(len(support))])/sum(support)
                    updated_score=float("{0:.3f}".format(updated_score))
                    #update_results.update_results('LSTM',self.dataset_map[self.train_set],updated_score)
                    print '**VAL %d, acc=%g, cost=%g, F1 score = %g'%(y_valid.shape[0],val_acc,val_cost,updated_score)

                    #print '**VAL %d, acc=%g, cost=%g, F1 score = %g' % (y_valid.shape[0], val_acc, val_cost,val_score)
                    valResult.append({'lr':lrate,'decay_rate':decay_rate,'epoch':i+1,'valAcc':val_acc,'valScore':updated_score})
                    if bestScore<updated_score:
                        bestScore = updated_score
                        bestModel = len(valResult)
                    #save model
                    save_path = saver.save(sess, model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(dataset,l,d,i+1))

hparams={'lr':valResult[bestModel-1]['lr'],'d':valResult[bestModel-1]['decay_rate'],'epoch':valResult[bestModel-1]['epoch']}
update_results.update_params('LSTM',dataset,hparams)
for vRes in valResult:
    print vRes
    
#check best model and apply on test model
##tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #get best model
    # l = valResult[bestModel-1]['lr']
    # dr = valResult[bestModel-1]['decay_rate']
    # i = valResult[bestModel-1]['epoch']
##    l = 0.00075
##    dr = 1
##    i = 120
    with open(PARAMS,'r') as res:
        params=json.load(res)
    lr,d,epoch=params['LSTM'][dataset]['lr'],params['LSTM'][dataset]['d'],params['LSTM'][dataset]['epoch']
    print 'lrate:',lr
    print 'decay rate',d
    print 'epochs',epoch
    saver.restore(sess, model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(dataset,lr,d,epoch+1))
    #save model
    #save_path = saver.save(sess, 'final_model/idcnn-lr_%g-dr_%g_ep%d.ckpt'%(l,dr,i))

    #evaluation the model on test set
    test_acc, test_cost,test_score,out_dict = testModule(X_test,y_test,final=True,plot_name=test_set)
    # print '**TEST RESULT:'
    # print '**TEST %d, acc=%g, cost=%g, F1 score = %g' % (y_test.shape[0], test_acc, test_cost,test_score)

    f1_scores=out_dict['f1-score'][:-1]
    support=out_dict['support'][:-1]
    updated_score=sum([f1_scores[i]*support[i] for i in range(len(support))])/sum(support)
    updated_score=float("{0:.3f}".format(updated_score))
    update_results.update_results('LSTM',test_set,updated_score)
    print "Updated score:",updated_score
    print '**TEST RESULT:'
    print '**TEST %d, acc=%g, cost=%g, F1 score = %g'%(y_test.shape[0],test_acc,test_cost,updated_score)
    print 'Updating the RESULTS file....'
            
