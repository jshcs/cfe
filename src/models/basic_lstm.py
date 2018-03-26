import tensorflow as tf
from config import *
from umass_parser import *
from features import *
from readDataset import *


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceLabelling:

    def __init__(self, data, target,num_hidden=128, num_layers=1,lrate = 1e-3):
        self.data = data
        self.target = target
##        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.lrate = lrate
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.BasicLSTMCell(self._num_hidden)
        output, state = tf.nn.dynamic_rnn(network, data, dtype=tf.float32)
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(
            self.target * tf.log(self.prediction), [1, 2])
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(self.lrate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


##def read_dataset():
##    dataset = sets.Ocr()
##    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
##    dataset['data'] = dataset.data.reshape(
##        dataset.data.shape[:-2] + (-1,)).astype(float)
##    train, test = sets.Split(0.66)(dataset)
##    return train, test


if __name__ == '__main__':
    length = config_params["max_stream_length"]
    num_features = len(config_params["feature_names"])
    num_classes = len(labels)+1
    epochs = config_params["epochs"]
    batch_size = config_params["batch_size"]
    lrate = config_params["lrate"]
    
##    train_token,train_label,val_token,val_label,test_token,test_label = read_dataset()
    train_token,train_label = read_dataset("train")
    val_token,val_label = read_dataset("dev")
    data = tf.placeholder(tf.float32, [None, length, num_features])
    target = tf.placeholder(tf.float32, [None, length, num_classes])
##    dropout = tf.placeholder(tf.float32,0.0)
    learning_rate = tf.constant(lrate)
    model = SequenceLabelling(data, target, learning_rate)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        batch = len(train_token)//batch_size
        if len(train_token)%batch_size>0:
            batch = batch+1
        print(epoch)
        for b in range(batch):
            if b==batch-1:
                token_batch = train_token[b*batch_size:]
                label_batch = train_label[b*batch_size:]
            else:
                token_batch = train_token[b*batch_size:(b+1)*batch_size]
                label_batch = lebel_batch[b*batch_size:(b+1)*batch_size]
            sess.run(model.optimize, {
                data: token_batch, target: label_batch})
        if (epoch+1)%20==0:
            error = sess.run(model.error, {
                data: val_token, target: val_label})
            print('Epoch {:2d} error on valid data {:3.1f}%'.format(epoch + 1, 100 * error))
