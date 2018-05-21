import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,precision_recall_fscore_support
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
import update_results
from utils import plot_confusion_matrix
import json
from sklearn.utils import shuffle

class LSTM_Model():
    def __init__(self,train_set,use_gpu=False,do_test=False):
        self.train_set=train_set
        self.X_train,self.y_train,self.X_valid,self.y_valid,self.X_test,self.y_test=None,None,None,None,None,None
        self.lrate=config_params["lrate"]
        self.num_units=config_params["num_units"]
        self.length=config_params["max_stream_length"]
        self.num_features=len(config_params["feature_names"])+EMD_SIZE-1
        self.num_classes=len(labels)+1
        self.epochs=config_params["epochs"]
        self.tr_batch_size=config_params["batch_size"]
        self.layer_num=config_params["num_layer"]
        self.decay_rate=config_params["lrate_decay"]
        self.max_grad_norm=5.0
        self.target_names=ALL_TAGS
        self.target_names.append('unknown')
        self.dataset_map={0:'umass',1:'combined',2:'heldout',3:'umass_heldout'}
        self.use_gpu=use_gpu
        self.do_test=do_test
        if self.use_gpu:
            self.device="/gpu:0"
        else:
            self.device="/cpu:0"

    def get_data(self):
        print "Unloading the dataset...."
        # data_zip=np.load('../../data/we_npy/combined_dataset.npz')
        if self.train_set==0:
            print "UMass dataset...."
            data_zip=np.load('../../data/we_npy_no_bio/umass_dataset.npz')
            self.X_train=data_zip['X_train.npy']
            self.y_train=data_zip['y_train.npy']
            self.X_valid=data_zip['X_valid.npy']
            self.y_valid=data_zip['y_valid.npy']
            self.X_test=data_zip['X_test.npy']
            self.y_test=data_zip['y_test.npy']
        elif self.train_set==1:
            print "Combined dataset...."
            data_zip=np.load('../../data/we_npy_no_bio/combined_dataset.npz')
            self.X_train=data_zip['combined_X_train.npy']
            self.y_train=data_zip['combined_y_train.npy']
            self.X_valid=data_zip['combined_X_valid.npy']
            self.y_valid=data_zip['combined_y_valid.npy']
            self.X_test=data_zip['combined_X_test.npy']
            self.y_test=data_zip['combined_y_test.npy']
        final_zip=np.load('../../data/we_npy_no_bio/final_test.npz')
        self.X_final_test=final_zip['X_test']
        self.y_final_test=final_zip['y_test']

        print "Loaded the dataset...."
        print self.X_train.shape,self.X_valid.shape,self.X_test.shape,self.y_train.shape,self.y_valid.shape,self.y_test.shape,self.X_final_test.shape,self.y_final_test.shape
        print "-"*40

    def make_model(self):

        with tf.device(self.device):
            self.lr = tf.placeholder(tf.float32, [])
            self.keep_prob = tf.placeholder(tf.float32, [])
            self.batch_size = tf.placeholder(tf.int32,[])
            self.wo = tf.Variable(tf.truncated_normal([self.num_units,self.num_classes]))
            self.bo = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))
            self.model_save_path='ckpt/lstm'

        #lstm cell

        def lstm_cell():
            self.cell = rnn.LSTMCell(self.num_units, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

        #lstm network
        def lstm(tokens):
            self.cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
            self.outputs, _ = tf.nn.dynamic_rnn(self.cell_fw,tokens, dtype=tf.float32)
            self.output = tf.reshape(self.outputs, [-1,self.num_units])
            return self.output

        #input placeholder
        with tf.device(self.device):
            with tf.variable_scope('Inputs'):
                self.data = tf.placeholder(tf.float32, shape=(None, self.length, self.num_features))
                self.target = tf.placeholder(tf.float32, shape=(None, self.length, self.num_classes))
                print "data shape",self.data.get_shape,"target shape",self.target.get_shape

        #lstm network to get the output
        lstm_output = lstm(self.data)

        #output of lstm network after last softmax layer
        with tf.device(self.device):
            with tf.variable_scope('outputs'):
                self.label_preds = tf.matmul(lstm_output, self.wo) + self.bo
                ##	print lstm_output.get_shape,wo.get_shape
                self.label_pred = tf.reshape(self.label_preds, [-1,self.length, self.num_classes])


        #check predict result against ground truth

        self.correct_prediction = tf.equal(tf.argmax(self.label_pred, 2),tf.argmax(self.target,2))
        #get accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #cost function
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.target, logits = self.label_pred))
        #grandient desent
        self.tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.tvars), self.max_grad_norm)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars),global_step=tf.train.get_or_create_global_step())
        self.saver=tf.train.Saver(max_to_keep=10)



    def test_epoch(self,data_x,data_y,final,plot_name):

        fetches = [self.accuracy, self.cost, self.label_pred]
        data_size = data_y.shape[0]
        X_batch, y_batch = data_x,data_y
        feed_dict = {self.data:data_x, self.target:data_y, self.batch_size:data_size, self.keep_prob:1.0}
        _accs, _costs, _pred = self.sess.run(fetches, feed_dict)
        #F1 result
        if final==True:
            np.save('../../data/lstm_test_result_'+plot_name+'.npy',_pred,allow_pickle=False)
        _pred = np.argmax(_pred, axis = 2)
        pred = np.reshape(_pred,(1,-1))
        pred = np.squeeze(pred)
        ground_truth = np.argmax(data_y, axis = 2)
        ground_truth = np.reshape(ground_truth,(1,-1))
        ground_truth = np.squeeze(ground_truth)
        print "pred:",pred,"ground:",ground_truth
        _scores = f1_score(ground_truth,pred, average='weighted')
        print('classification report')
        print(classification_report(ground_truth,pred,target_names=self.target_names))

        clf_rep=precision_recall_fscore_support(ground_truth,pred)
        out_dict={
            "precision":clf_rep[0].round(2)
            ,"recall":clf_rep[1].round(2)
            ,"f1-score":clf_rep[2].round(2)
            ,"support":clf_rep[3]
        }
        if final==True:
            #np.set_printoptions(precision=2)
            cm=confusion_matrix(ground_truth,pred)
            print 'cm shape',cm.shape
            # print 'cm shape',cm.shape
            cm=cm[:len(labels),:len(labels)]
            print 'cm shape',cm.shape

            plt.figure()
            plot_confusion_matrix(cm,classes=self.target_names[:-1],
                                  title='LSTM consusion matrix w/o normalization')
            plt.savefig('lstm_cm_no_norm_'+plot_name+'.png')

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cm,classes=self.target_names[:-1],normalize=True,
                                  title='LSTM confusion matrix')

            plt.savefig('lstm_cm_norm_'+plot_name+'.png')

        return _accs, _costs, _scores,out_dict

    def train(self):
        print "*"*30
        print "Starting training...."
        tr_batch_size = 100
        display_num = 10
        decay_num = 20
        tr_batch_num = int(self.y_train.shape[0] / self.tr_batch_size)

        self.all_results=[]
        max_f1=0
        self.bestModel=0
        for lrate in LR_RANGE:
            for decay_rate in DECAY_RATE:
                self.sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                self.sess.run(tf.global_variables_initializer())
                for epoch in xrange(self.epochs):
                    _costs = 0.0
                    _accs = 0.0
                    _lr = lrate*(decay_rate**(epoch/decay_num))
                    self.X_train,self.y_train=shuffle(self.X_train,self.y_train,random_state=0)
                    for batch in xrange(tr_batch_num):
                        fetches = [self.accuracy, self.cost, self.train_op]
                        X_batch = self.X_train[batch*self.tr_batch_size:(batch+1)*self.tr_batch_size,:,:]
                        y_batch = self.y_train[batch*self.tr_batch_size:(batch+1)*self.tr_batch_size,:,:]
                        feed_dict = {self.data:X_batch, self.target:y_batch, self.batch_size:self.tr_batch_size,self.lr:_lr, self.keep_prob:1}
                        _acc, _cost, _ = self.sess.run(fetches, feed_dict)
                        _accs += _acc
                        _costs += _cost
                    mean_acc = _accs / tr_batch_num
                    mean_cost = _costs / tr_batch_num
                    if (epoch+1)%display_num==0:
                        print 'learning rate:',lrate,'decay_rate:',decay_rate
                        print 'epoch',epoch+1
                        print 'training %d, acc=%g, cost=%g '%(self.y_train.shape[0],mean_acc,mean_cost)
                    if (epoch+1)>=self.epochs:
                        print '**VAL RESULT:'
                        val_acc,val_cost,val_score,out_dict=self.test_epoch(self.X_valid,self.y_valid,final=False,plot_name=self.dataset_map[self.train_set])
            f1_scores=out_dict['f1-score'][:-1]
            support=out_dict['support'][:-1]
            updated_score=sum([f1_scores[i]*support[i] for i in range(len(support))])/sum(support)
            updated_score=float("{0:.3f}".format(updated_score))
            #update_results.update_results('LSTM',self.dataset_map[self.train_set],updated_score)
            print '**VAL %d, acc=%g, cost=%g, F1 score = %g'%(self.y_valid.shape[0],val_acc,val_cost,updated_score)
            self.all_results.append({'lr':lrate,'decay_rate':decay_rate,'epoch':epoch+1,'valAcc':val_acc,'valScore':updated_score})
            if max_f1<updated_score:
                max_f1=updated_score
                self.bestModel=len(self.all_results)
            #save model
            save_path=self.saver.save(self.sess,self.model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(self.dataset_map[self.train_set],lrate,decay_rate,epoch+1))

        hparams={'lr':self.all_results[self.bestModel-1]['lr'],'d':self.all_results[self.bestModel-1]['decay_rate'],'epoch':self.all_results[self.bestModel-1]['epoch']}
        update_results.update_params('LSTM',self.dataset_map[self.train_set],hparams)

    def test_on_testset(self,data_x,data_y,dataset):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as self.sess:
            with open(PARAMS,'r') as res:
                params=json.load(res)
            print "Dataset:",self.dataset_map[dataset]
            if dataset==3:
                idx=0
            elif dataset==2:
                idx=1
            else:
                idx=dataset
            lr,d,epoch=params['LSTM'][self.dataset_map[idx]]['lr'],params['LSTM'][self.dataset_map[idx]]['d'],params['LSTM'][self.dataset_map[idx]]['epoch']
            print 'lrate:',lr
            print 'decay rate',d
            print 'epochs',epoch
            if dataset==2:
                self.saver.restore(self.sess,self.model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(self.dataset_map[1],lr,d,epoch))
            elif dataset==3:
                self.saver.restore(self.sess,self.model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(self.dataset_map[0],lr,d,epoch))
            else:
                self.saver.restore(self.sess,self.model_save_path+'-mod_%s-lr_%g-dr_%g_ep%d.ckpt'%(self.dataset_map[dataset],lr,d,epoch))
            #evaluation the model on final test set
            test_acc,test_cost,test_score,out_dict=self.test_epoch(data_x,data_y,final=True,plot_name=self.dataset_map[dataset])
            #print '**TEST RESULT:'
            #print '**TEST %d, acc=%g, cost=%g, F1 score = %g'%(data_y.shape[0],test_acc,test_cost,test_score)
            #print 'Updating the RESULTS file....'
            f1_scores=out_dict['f1-score'][:-1]
            support=out_dict['support'][:-1]
            updated_score=sum([f1_scores[i]*support[i] for i in range(len(support))])/sum(support)
            updated_score=float("{0:.3f}".format(updated_score))
            update_results.update_results('LSTM',self.dataset_map[dataset],updated_score)
            print "Updated score:",updated_score
            print '**TEST RESULT:'
            print '**TEST %d, acc=%g, cost=%g, F1 score = %g'%(data_y.shape[0],test_acc,test_cost,updated_score)
            print 'Updating the RESULTS file....'


    def predict(self,data_x):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as self.sess:
            with open(PARAMS,'r') as res:
                params=json.load(res)
            lr,d,epoch=params['LSTM']['lr'],params['LSTM']['d'],params['LSTM']['epoch']
            print 'lrate:',lr
            print 'decay rate',d
            print 'epochs',epoch
            self.saver.restore(self.sess,self.model_save_path+'-lr_%g-dr_%g_ep%d.ckpt'%(lr,d,epoch))
            #evaluation the model on test set
            fetches=[self.label_pred]
            data_size=data_x.shape[0]
            #X_batch,y_batch=data_x,data_y
            feed_dict={self.data:data_x,self.batch_size:data_size,self.keep_prob:1.0}
            _pred=self.sess.run(fetches,feed_dict)
            #F1 result
            print _pred
            print np.array(_pred).shape
            _pred=np.argmax(_pred,axis=2)
            print _pred
            print _pred.shape
            pred=np.reshape(_pred,(1,-1))
            print _pred
            print _pred.shape
            pred=np.squeeze(pred)
            print pred
            print pred.shape

    def run_lstm(self):
        self.get_data()
        self.make_model()

        if self.do_test==False:
            self.train()
            self.test_on_testset(self.X_test,self.y_test,self.train_set)
        else:
            self.test_on_testset(self.X_final_test,self.y_final_test,2)
            # self.test_on_testset(self.X_test,self.y_test,0)
        # get_sample_citations()
        # self.predict(self.X_test[:5,:,:])




model=LSTM_Model(1,use_gpu=False,do_test=True)
model.run_lstm()

