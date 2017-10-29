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


class C_MODEL(object):
    """ 
    """

    def __init__(self, config, graph):
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
        self.normer = Norm()
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
        self.__D_steps = 0
        # data
        self.__G_samples = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, self.normer.PLAYERS, self.normer.COLS, self.normer.ROWS], name='G_samples')
        self.__X = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, self.normer.PLAYERS, self.normer.COLS, self.normer.ROWS], name='real_data')
        # adversarial learning : wgan
        self.__build_wgan()

        # summary
        self.__summary_D_op = tf.summary.merge(tf.get_collection('D'))
        self.__summary_D_valid_op = tf.summary.merge(
            tf.get_collection('D_valid'))
        self.D_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'D')
        self.D_valid_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'D_valid')

    def __build_wgan(self):
        with tf.name_scope('WGAN'):
            D_real = self.inference(self.__X, seq_len=None)
            self.__D_fake = self.inference(
                self.__G_samples, seq_len=None, reuse=True)
            # loss function
            self.__D_loss, F_real, F_fake, grad_pen = self.__D_loss_fn(
                self.__X, self.__G_samples, self.__D_fake, D_real, self.penalty_lambda)
            theta_D = self.__get_var_list()
            with tf.name_scope('D_optimizer') as scope:
                D_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                D_grads = tf.gradients(self.__D_loss, theta_D)
                D_grads = list(zip(D_grads, theta_D))
                self.__D_train_op = D_optimizer.apply_gradients(
                    grads_and_vars=D_grads, global_step=self.__global_steps)
            # logging
            for grad, var in D_grads:
                self.__summarize(var.name, grad, collections='D',
                                 postfix='gradient')
            tf.summary.scalar('D_loss', self.__D_loss,
                              collections=['D', 'D_valid'])
            tf.summary.scalar('F_real', F_real, collections=['D'])
            tf.summary.scalar('F_fake', F_fake, collections=['D'])
            tf.summary.scalar('grad_pen', grad_pen, collections=['D'])

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
        theta_D = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('D'):
                theta_D.append(v)
                self.__summarize(v.op.name, v, collections='D',
                                 postfix='Trainable')
        return theta_D

    def __leaky_relu(self, features, alpha=0.7):
        return tf.maximum(features, alpha * features)

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                            forget_bias=1.0, state_is_tuple=True,
                            # activation=self.__leaky_relu, cell_clip=2,
                            activation=tf.nn.tanh, reuse=tf.get_variable_scope().reuse)

    def inference(self, inputs, seq_len=None, reuse=False):
        """
        Inputs
        ------
        inputs : float, shape=[batch_size, seq_length=100, PLAYERS=11, COLS=98, ROWS=46]
            real(from data) or fake(from G)
        seq_len : 
            temparily not used

        Return
        ------
        decision : bool
            real(from data) or fake(from G)
        """
        with tf.variable_scope('D', reuse=reuse) as scope:
            # unstack, axis=1 -> [batch, time, feature]
            print(inputs)
            inputs = tf.transpose(inputs, perm=[0, 1, 3, 4, 2])
            print(inputs)
            inputs = tf.unstack(inputs, num=self.seq_length, axis=1)
            blstm_input = []
            output_list = []
            for time_step in range(self.seq_length):
                with tf.variable_scope('conv') as scope:
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    filters_list = [32, 64, 128, 256]
                    next_input = inputs[time_step]
                    for i in range(4):
                        with tf.variable_scope('conv' + str(i)) as scope:
                            conv = layers.conv2d(
                                inputs=next_input,
                                num_outputs=filters_list[i],
                                kernel_size=[5, 5],
                                stride=2,
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                weights_initializer=layers.xavier_initializer(
                                    uniform=False),
                                weights_regularizer=None,
                                biases_initializer=tf.zeros_initializer(),
                                reuse=scope.reuse,
                                scope=scope
                            )
                            next_input = conv
                    with tf.variable_scope('fc') as scope:
                        flat_input = layers.flatten(next_input)
                        fc = layers.fully_connected(
                            inputs=flat_input,
                            num_outputs=self.hidden_size,
                            activation_fn=tf.nn.relu,
                            weights_initializer=layers.xavier_initializer(
                                uniform=False),
                            biases_initializer=tf.zeros_initializer(),
                            reuse=scope.reuse,
                            scope=scope
                        )
                        blstm_input.append(fc)
            with tf.variable_scope('stack_blstm') as scope:
                stack_blstm, _, _ = rnn.stack_bidirectional_rnn(
                    cells_fw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    cells_bw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    inputs=blstm_input,
                    dtype=tf.float32,
                    sequence_length=seq_len
                )
            with tf.variable_scope('output') as scope:
                for i, out_blstm in enumerate(stack_blstm):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('fc') as scope:
                        fc = layers.fully_connected(
                            inputs=out_blstm,
                            num_outputs=1,
                            activation_fn=self.__leaky_relu,
                            weights_initializer=layers.xavier_initializer(
                                uniform=False),
                            biases_initializer=tf.zeros_initializer(),
                            reuse=scope.reuse,
                            scope=scope
                        )
                        output_list.append(fc)
            # stack, axis=1 -> [batch, time, feature]
            decisions = tf.stack(output_list, axis=1)
            print('decisions', decisions)
            decision = tf.reduce_mean(decisions, axis=1)
            print('decision', decision)
            return decision

    def __D_loss_fn(self, __X, __G_sample, D_fake, D_real, penalty_lambda):
        """ D loss
        """
        with tf.name_scope('D_loss') as scope:
            # grad_pen, base on paper (Improved WGAN)
            epsilon = tf.random_uniform(
                [self.batch_size, 1, 1, 1, 1], minval=0.0, maxval=1.0)
            __X_inter = epsilon * __X + (1.0 - epsilon) * __G_sample
            grad = tf.gradients(
                self.inference(__X_inter, seq_len=None, reuse=True), [__X_inter])[0]
            sum_ = tf.reduce_sum(tf.square(grad), axis=[1, 2])
            grad_norm = tf.sqrt(sum_)
            grad_pen = penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            f_fake = tf.reduce_mean(D_fake)
            f_real = tf.reduce_mean(D_real)

            loss = f_fake - f_real + grad_pen
        return loss, f_real, f_fake, grad_pen

    def step(self, sess, G_samples, real_data):
        """ train one batch on D
        """
        self.__D_steps += 1
        feed_dict = {self.__G_samples: G_samples,
                     self.__X: real_data}
        loss, global_steps, _ = sess.run(
            [self.__D_loss, self.__global_steps, self.__D_train_op], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__D_steps % 500 == 0:  # % 500 to save space
            summary = sess.run(self.__summary_D_op, feed_dict=feed_dict)
            # log
            self.D_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss, global_steps

    def D_log_valid_loss(self, sess, G_samples, real_data):
        """ one batch valid loss
        """
        feed_dict = {self.__G_samples: G_samples,
                     self.__X: real_data}
        loss, global_steps = sess.run(
            [self.__D_loss, self.__global_steps], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__D_steps % 500 == 0:  # % 500 to save space
            summary = sess.run(self.__summary_D_valid_op, feed_dict=feed_dict)
            # log
            self.D_valid_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss

    def evaluate(self, sess, G_samples):
        """ evaluate G_sample by Critic
        """
        # feed_dict = {self.__G_samples: G_samples}
        # rewards = sess.run(self.__D_fake, feed_dict=feed_dict)
        return self.__D_fake
