from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DCAEModel(object):
    """ Model of Denoising Convolutional AutoEncoder for data imputation
    # TODO list
        * weights/ bias initilizer
        * filter amount
        * multi-gpu
        * self.is_training 
        * predict and visulize
    """

    def __init__(self, config, graph):
        """ build up the whole tensorflow computational graph
        Params
        ------
        config : class, hyper-perameters
            * filter_numbers : int, list, the number of filters for each conv and decov in reversed order
            * filter_strides : int, list, the value of stride for each conv and decov in reversed order
            * batch_size : int, mini batch size
            * log_dir : string, the path to save training summary
            * learning_rate : float, adam's learning rate
            * input_shape : int, list, e.g. [?, nums_VD, nums_Interval, nums_features]
        graph : tensorflow default graph
            for summary writer and __global_step init
        """
        # hyper-parameters
        self.filter_numbers = config.filter_numbers
        self.filter_strides = config.filter_strides
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.input_shape = config.input_shape
        # steps
        self.__global_step = tf.train.get_or_create_global_step(graph=graph)
        # data
        self.__corrupt_data = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3]], name='corrupt_data')
        self.__raw_data = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.input_shape[1], self.input_shape[2], 3], name='raw_data')
        # model
        self.__logits = self.__inference(
            self.__corrupt_data, self.filter_numbers, self.filter_strides)
        self.__loss = self.__loss_function(
            self.__logits, self.__raw_data)
        # add to summary
        tf.summary.scalar('loss', self.__loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        # print(tf.trainable_variables())
        self.__train_op = optimizer.minimize(
            self.__loss, global_step=self.__global_step)

        # summary
        self.__merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'train', graph=graph)

    def __inference(self, __corrupt_data, filter_numbers, filter_strides):
        """ construct the AutoEncoder model
        Params
        ------
        __corrupt_data : placeholder, shape=[batch_size, nums_vd, nums_interval, features]
            get by randomly corrupting raw data
        filter_numbers : int, list
            the number of filters for each conv and decov in reversed order. e.g. [32, 64, 128]
        filter_strides : int, list, 
            the value of stride for each conv and decov in reversed order. e.g. [1, 2, 2]
        
        Return
        ------
        output : the result of AutoEndoer, shape is same as '__corrupt_data'
        """
        def lrelu(x, alpha=0.3):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
        print("__corrupt_data:", __corrupt_data)
        shapes_list = []
        # encoder
        current_input = __corrupt_data
        for layer_id, out_filter_amount in enumerate(filter_numbers):
            with tf.variable_scope('conv' + str(layer_id)) as scope:
                # shape
                shapes_list.append(current_input.get_shape().as_list())
                in_filter_amount = current_input.get_shape().as_list()[3]
                # init
                kernel_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                bias_init = tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                W = tf.get_variable(name='weights', shape=[
                                    3, 3, in_filter_amount, out_filter_amount], initializer=kernel_init)
                b = tf.get_variable(
                    name='bias', shape=out_filter_amount, initializer=bias_init)
                # conv
                stide = filter_strides[layer_id]
                output = lrelu(
                    tf.add(tf.nn.conv2d(
                        input=current_input, filter=W, strides=[1, stide, stide, 1], padding='SAME'), b))
                current_input = output
                print(scope.name, output)

        # print('shapes_list:', shapes_list)
        # reverse order for decoder part
        shapes_list.reverse()
        filter_strides.reverse()

        # decoder
        for layer_id, layer_shape in enumerate(shapes_list):
            with tf.variable_scope('deconv' + str(layer_id)) as scope:
                # shape
                in_filter_amount = current_input.get_shape().as_list()[3]
                if layer_id == len(shapes_list)-1:
                    out_filter_amount = 3 # only regress 3 dims as [d, f, s] 
                else:
                    out_filter_amount = layer_shape[3]               
                # init
                kernel_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                bias_init = tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                W = tf.get_variable(name='weights', shape=[
                                    3, 3, out_filter_amount, in_filter_amount], initializer=kernel_init)
                b = tf.get_variable(
                    name='bias', shape=out_filter_amount, initializer=bias_init)
                # deconv
                stide = filter_strides[layer_id]
                output = lrelu(
                    tf.add(tf.nn.conv2d_transpose(
                        value=current_input, filter=W, output_shape=tf.stack([layer_shape[0], layer_shape[1], layer_shape[2], out_filter_amount]), strides=[1, stide, stide, 1], padding='SAME'), b))
                current_input = output
                print(scope.name, output)

        return output

    def __loss_function(self, __logits, labels):
        """ l2 function (mean square error)
        Params
        ------
        __logits : tensor, from __inference()
        labels : placeholder, raw data

        Return
        ------
        l2_mean_loss : tensor 
            MSE of one batch
        """
        with tf.name_scope('l2_loss'):
            vd_losses = tf.squared_difference(__logits, labels)
            l2_mean_loss = tf.reduce_mean(vd_losses)
        print('l2_mean_loss:', l2_mean_loss)
        return l2_mean_loss

    def step(self, sess, inputs, labels):
        """ train one batch and update one time
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]
        labels: raw data, shape=[batch_size, nums_vd, nums_interval, features]

        Return
        ------
        loss : float 
            MSE of one batch
        global_steps : 
            the number of batches have been trained
        """
        feed_dict = {self.__corrupt_data: inputs, self.__raw_data: labels}
        summary, loss, global_steps, _ = sess.run(
            [self.__merged_op, self.__loss, self.__global_step, self.__train_op], feed_dict=feed_dict)
        # write summary
        self.train_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss, global_steps

    def compute_loss(self, sess, inputs, labels):
        """ compute loss
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]
        labels: raw data, shape=[batch_size, nums_vd, nums_interval, features]

        Return
        ------
        loss : float 
            MSE of one batch
        """
        feed_dict = {self.__corrupt_data: inputs, self.__raw_data: labels}
        loss = sess.run(self.__loss, feed_dict=feed_dict)
        return loss

    def predict(self, sess, inputs):
        """ recover the inputs (corrupted data)
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]
        
        Return
        ------
        result : raw data, shape=[batch_size, nums_vd, nums_interval, features]
        """
        feed_dict = {self.__corrupt_data: inputs}
        result = sess.run(self.__logits, feed_dict=feed_dict)
        return result


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.filter_numbers = [32, 64, 128]
        self.filter_strides = [1, 2, 2]
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.log_dir = "test_log_dir/"
        self.input_shape = [256, 100, 12, 5]


def test():
    with tf.Graph().as_default() as g:
        config = TestingConfig()
        model = DCAEModel(config, graph=g)
        # train
        X = np.zeros(shape=[256, 100, 12, 5])
        Y = np.zeros(shape=[256, 100, 12, 3])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(config.total_epoches):
                loss, global_steps = model.step(sess, X, Y)
                print('loss', loss)


if __name__ == '__main__':
    test()
