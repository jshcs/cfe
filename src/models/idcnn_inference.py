import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from cnn_config import *
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#loading data
with open('../../data/train.pkl', 'rb') as inp:
    X_train = pickle.load(inp)
    y_train = pickle.load(inp)

with open('../../data/val.pkl', 'rb') as inp:
    X_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

with open('../../data/test.pickle', 'rb') as inp:
    X_test = pickle.load(inp)
    y_test = pickle.load(inp)

lrate = config_params["lrate"]
length = config_params["max_stream_length"]
num_features = len(config_params["feature_names"])
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
target_names.append('unknown')

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

#test on other data set (valid or test)
def testModule(data_x,data_y):
    fetches = [accuracy, cost, idcnn_output]
    data_size = data_y.shape[0]
    X_batch, y_batch = data_x,data_y
    feed_dict = {data:data_x, target:data_y, dropout:1.0}
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

#begin to train
batch_size = 20
display_num = 10
decay_num = 20
tr_batch_num = y_train.shape[0]/batch_size
if y_train.shape[0]%batch_size>0:
    tr_batch_num = tr_batch_num+1
saver = tf.train.Saver(max_to_keep=100)

valResult = []
bestScore = 0.0

for l in np.arange(lrate,lrate*2,15e-5):
    for d in np.arange(decay_rate,1.01,0.15):
        with tf.Session() as sess:
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
                if (i+1)>=100 and (i+1)%10==0:
                    print '**VAL RESULT:'
                    val_acc, val_cost,val_score = testModule(X_valid,y_valid)
                    print '**VAL %d, acc=%g, cost=%g, F1 score = %g' % (y_valid.shape[0], val_acc, val_cost,val_score)
                    valResult.append({'lr':l,'decay_rate':d,'epoch':i+1,'valAcc':val_acc,'valScore':val_score})
                    if bestScore<val_score:
                        bestScore = val_score
                        bestModel = len(valResult)
                    #save model
                    save_path = saver.save(sess, model_save_path+'-lr_%g-dr_%g_ep%d.ckpt'%(l,d,i+1))

for vRes in valResult:
    print vRes
    
#check best model and apply on test model
##tf.reset_default_graph()
with tf.Session() as sess:
    #get best model
    l = valResult[bestModel-1]['lr']
    dr = valResult[bestModel-1]['decay_rate']
    i = valResult[bestModel-1]['epoch']
    saver.restore(sess, model_save_path+'-lr_%g-dr_%g_ep%d.ckpt'%(l,d,i))
    #evaluation the model on test set
    test_acc, test_cost,test_score = testModule(X_test,y_test)
    print '**TEST RESULT:'
    print '**TEST %d, acc=%g, cost=%g, F1 score = %g' % (y_test.shape[0], test_acc, test_cost,test_score)

            
