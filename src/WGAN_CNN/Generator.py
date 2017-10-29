"""
modeling
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from utils_cnn import Norm


class G_MODEL(object):
    """
    """

    def __init__(self, config, critic_inference, graph):
        """ TO build up the graph
        Inputs
        ------
        config : 
            * batch_size : mini batch size
            * log_dir : path to save training summary
            * learning_rate : adam's learning rate
            * hidden_size : number of hidden units in LSTM
            * rnn_layers : number of stacked LSTM 
            * seq_length : length of LSTM
            * num_features : dimensions of input feature
            * latent_dims : dimensions of latent feature
            * penalty_lambda = gradient penalty's weight, ref from  paper of 'improved-wgan'
        graph : 
            tensorflow default graph
        """
        # hyper-parameters
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.hidden_size = config.hidden_size
        self.rnn_layers = config.rnn_layers
        self.seq_length = config.seq_length
        self.num_features = config.num_features
        self.latent_dims = config.latent_dims
        self.penalty_lambda = config.penalty_lambda
        self.if_log_histogram = config.if_log_histogram
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.__G_steps = 0
        # IO
        self.critic = critic_inference
        self.__z = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.num_features], name='latent_input')
        self.__X = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, self.num_features], name='real_data')
        # adversarial learning : wgan
        self.__build_wgan()

        # summary
        self.__summary_G_op = tf.summary.merge(tf.get_collection('G'))
        self.G_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G')

    def __build_wgan(self):
        with tf.name_scope('WGAN'):
            self.__G_sample = self.__G(self.__z, seq_len=None)
            # loss function
            self.__G_loss = self.__G_loss_fn(self.__G_sample)
            with tf.name_scope('G_optimizer') as scope:
                theta_G = self.__get_var_list()
                G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                G_grads = tf.gradients(self.__G_loss, theta_G)
                G_grads = list(zip(G_grads, theta_G))
                self.__G_train_op = G_optimizer.apply_gradients(
                    grads_and_vars=G_grads, global_step=self.__global_steps)
            # logging
            for grad, var in G_grads:
                self.__summarize(var.name, grad, collections='G',
                                 postfix='gradient')
            tf.summary.scalar('G_loss', self.__G_loss, collections=['G'])

    def __summarize(self, name, value, collections, postfix=''):
        """ Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args
        ----
        name : string
        value : Tensor
        collections : list of string
        postfix : string
        Returns
        -------
            nothing
        """
        if self.if_log_histogram:
            tensor_name = name + '/' + postfix
            tf.summary.histogram(tensor_name,
                                 value, collections=collections)
            # tf.summary.scalar(tensor_name + '/sparsity',
            #                   tf.nn.zero_fraction(x), collections=collections)

    def __get_var_list(self):
        """ to get both Generator's and Discriminator's trainable variables
        and add trainable variables into histogram
        """
        trainable_V = tf.trainable_variables()
        theta_G = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('G'):
                theta_G.append(v)
                self.__summarize(v.op.name, v, collections='G',
                                 postfix='Trainable')
        return theta_G

    def __leaky_relu(self, features, alpha=0.7):
        return tf.maximum(features, alpha * features)

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                            forget_bias=1.0, state_is_tuple=True, num_proj=6 * 3 * 512,  # TODO !!!
                            activation=tf.nn.tanh, reuse=tf.get_variable_scope().reuse)

    def __G(self, inputs, seq_len=None, if_pretrain=False):
        """ TODO
        Inputs
        ------
        inputs : float, shape=[batch, dims]
            latent variables
        seq_len : 
            temparily not used

        Return
        ------
        result : float, shape=[batch, length, 23]
            generative result (script)
        """
        with tf.variable_scope('G'):
            # init
            cell = rnn.MultiRNNCell(
                [self.__lstm_cell() for _ in range(self.rnn_layers)])
            state = cell.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            # model
            output_list = []
            for time_step in range(self.seq_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('stack_lstm') as scope:
                    stack_lstm, _ = cell(
                        inputs=inputs, state=state, scope=scope)
                    self.__summarize('stack_lstm', stack_lstm, collections=[
                        'G'], postfix='Activation')
                with tf.name_scope('deconv'):
                    linear_reshape = tf.reshape(
                        stack_lstm, shape=[self.batch_size, 6, 3, 512])
                    filters_list = [128, 64, 11]
                    next_input = linear_reshape
                    for i in range(len(filters_list)):
                        with tf.variable_scope('deconv' + str(i)) as scope:
                            deconv = layers.conv2d_transpose(
                                inputs=next_input,
                                num_outputs=filters_list[i],
                                kernel_size=[5, 5],
                                stride=2,
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer(
                                    uniform=False),
                                biases_initializer=tf.zeros_initializer(),
                                reuse=scope.reuse,
                                scope=scope
                            )
                            next_input = deconv
                with tf.name_scope('softmax'):
                    reshape_out = tf.reshape(
                        next_input, shape=[-1, Norm.COLS * Norm.ROWS, Norm.PLAYERS])
                    softmax_out = tf.nn.softmax(reshape_out, dim=1)
                    reshape_softmax_out = tf.reshape(
                        softmax_out, shape=[-1, Norm.COLS, Norm.ROWS, Norm.PLAYERS])
                    transpose_result = tf.transpose(
                        reshape_softmax_out, perm=[0, 3, 1, 2])
                output_list.append(transpose_result)
            # stack, axis=1 -> [batch, time, feature]
            result = tf.stack(output_list, axis=1)
            print('result', result)
            return result

    def __G_loss_fn(self, fake_samples):
        """ G loss
        """
        with tf.name_scope('G_loss') as scope:
            loss = -tf.reduce_mean(self.critic(fake_samples, reuse=True))
        return loss

    def step(self, sess, latent_inputs):
        """ train one batch on G
        """
        self.__G_steps += 1
        feed_dict = {self.__z: latent_inputs}
        loss, global_steps, _ = sess.run(
            [self.__G_loss, self.__global_steps,
                self.__G_train_op], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__G_steps % 100 == 0:  # % 100 to save space
            summary = sess.run(self.__summary_G_op, feed_dict=feed_dict)
            # log
            self.G_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss, global_steps

    def generate(self, sess, latent_inputs):
        """ to generate result
        """
        feed_dict = {self.__z: latent_inputs}
        result = sess.run(self.__G_sample, feed_dict=feed_dict)
        return result
