import tensorflow as tf
from config import *
from umass_parser import *
from features import *

def readData(dataset_type):
        if dataset_type=='train':
		data_obj=GetDict(TRAIN_FILE)
	elif dataset_type=='test':
		data_obj=GetDict(TEST_FILE)
	elif dataset_type=='dev':
		data_obj=GetDict(DEV_FILE)

        data_obj.make_dict()
	inputString = data_obj.citation_strings
        outputLabels = []
        inputTokens = []
        
        for s in inputString:
                temp = []
                inputTokens.append(data_obj.token_label[s][0])
                for l in data_obj.token_label[s][1]:
                        oneHot = [0]*(len(labels)+1)
                        if l in labels:
                                oneHot[labels[l]] = 1
                                temp.append(oneHot)
                        else:
                                oneHot[6] = 1
                                temp.append(oneHot)
                while len(temp)<config_params["max_stream_length"]:
                        oneHot = [0]*(len(labels)+1)
                        oneHot[6] = 1
                        temp.append(oneHot)
                outputLabels.append(temp)
        
        return inputString,inputTokens,outputLabels

def getFeatres(tokens):
        features = []
        for cString in tokens:
                temp = []
                for w in cString:
                        wordFeature = Features(w)
                        if len(w)<1:
                                f = [None]*len(config_params['feature_names'])
                        else:
                                f = wordFeature.vectorize()
                        temp.append(f)
                while len(temp)<config_params["max_stream_length"]:
                        f = [None]*len(config_params['feature_names'])
                        temp.append(f)
                features.append(temp)

        return features

inputString,inputTokens,outputLabels = readData('train')
featureTokens = getFeatres(inputTokens)

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([config_params["num_units"],
                                          len(labels)+1]))
out_bias=tf.Variable(tf.random_normal([len(labels)+1]))

#defining placeholders
#input tokens placeholder
x=tf.placeholder(tf.int32,[None,config_params["max_stream_length"],
                           len(config_params['feature_names'])])
#input label placeholder
y=tf.placeholder(tf.int32,[None,config_params["max_stream_length"],
                           len(labels)+1])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
inputToken=tf.unstack(x,config_params["max_stream_length"],1)

#defining the network
lstm_layer=tf.contrib.rnn.BasicLSTMCell(config_params["num_units"],forget_bias=1)
outputs,_=tf.contrib.rnn.static_rnn(lstm_layer,inputToken,dtype=tf.int32)

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs,out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=config_params["lrate"]).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,2),tf.argmax(y,2))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<=config_params["epochs"]:
        batch_x = tf.Variable(featureTokens)
        batch_y = tf.Variable(outputLabels)

        batch_x=batch_x.reshape((len(inputString),config_params["max_stream_length"],
                                 len(config_params['feature_names'])))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1


##
##class BaseModel():
##	def __init__(self):
##		self.config=config_params
##		self.sess=None
##	def global_init_variables(self):
##		init=tf.global_variables_initializer()
##		self.sess.run(init)
