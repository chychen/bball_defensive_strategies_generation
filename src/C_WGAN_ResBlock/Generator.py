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
from utils import DataFactory
import Libs as libs


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
        self.latent_dims = config.latent_dims
        self.penalty_lambda = config.penalty_lambda
        self.latent_penalty_lambda = config.latent_penalty_lambda
        self.if_log_histogram = config.if_log_histogram
        self.n_resblock = config.n_resblock
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.__G_steps = 0
        # IO
        self.critic = critic_inference
        self.__z = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.latent_dims], name='latent_input')
        self.__cond = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, 13], name='team_a')
        # adversarial learning : wgan
        self.__build_wgan()

        # summary
        self.__summary_G_op = tf.summary.merge(tf.get_collection('G'))
        self.__summary_G_weight_op = tf.summary.merge(
            tf.get_collection('G_weight'))
        self.G_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G', graph=graph)

    def __build_wgan(self):
        with tf.name_scope('WGAN'):
            self.__G_sample = self.__G(self.__z, self.__cond, seq_len=None)
            # loss function
            self.__G_loss, self.__G_penalty_latents_w = self.__G_loss_fn(
                self.__G_sample, self.__cond, lambda_=self.latent_penalty_lambda)
            with tf.name_scope('G_optimizer') as scope:
                theta_G = self.__get_var_list('G')
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
            tf.summary.scalar('G_penalty_latents_w',
                              self.__G_penalty_latents_w, collections=['G'])

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

    def __get_var_list(self, prefix):
        """ to get both Generator's trainable variables and add trainable variables into histogram
        """
        trainable_V = tf.trainable_variables()
        theta = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith(prefix):
                theta.append(v)
                self.__summarize(v.op.name, v, collections=prefix,
                                 postfix='Trainable')
        return theta

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=False, initializer=None,
                            forget_bias=1.0, state_is_tuple=True,
                            activation=tf.nn.tanh, reuse=tf.get_variable_scope().reuse)

    def __G(self, latents, conds, seq_len=None, if_pretrain=False):
        """ TODO
        Inputs
        ------
        latents : float, shape=[batch, length=100, dims=10]
            latent variables
        conds : float, shape=[batch, length=100, dims=13]
            conditional constraints of team A and ball
        seq_len : 
            temparily not used

        Return
        ------
        result : float, shape=[batch, length=100, 23]
            generative result (script)
        """
        with tf.variable_scope('G'):  # init
            # concat_ = tf.concat([conds, latents], axis=-1)
            # with tf.variable_scope('linear') as scope:
            #     linear = layers.fully_connected(
            #         inputs=concat_,
            #         num_outputs=256,
            #         activation_fn=libs.leaky_relu,
            #         weights_initializer=layers.xavier_initializer(
            #             uniform=False),
            #         biases_initializer=tf.zeros_initializer(),
            #         scope=scope
            #     )
            #     print(linear)
            with tf.variable_scope('conds_linear') as scope:
                conds_linear = layers.fully_connected(
                    inputs=conds,
                    num_outputs=256,
                    activation_fn=None,
                    weights_initializer=layers.xavier_initializer(
                        uniform=False),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope
                )
            with tf.variable_scope('latents_linear') as scope:
                latents_linear = layers.fully_connected(
                    inputs=latents,
                    num_outputs=256,
                    activation_fn=None,
                    weights_initializer=layers.xavier_initializer(
                        uniform=False),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope
                )
            latents_linear = tf.reshape(latents_linear, shape=[
                                        self.batch_size, 1, 256])
            next_input = tf.add(conds_linear, latents_linear)
            print(next_input)
            # residual block
            for i in range(self.n_resblock):
                res_block = libs.residual_block('Res' + str(i), next_input)
                next_input = res_block
                print(next_input)
            with tf.variable_scope('conv_result') as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = libs.leaky_relu(normed)
                conv_result = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=10,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=libs.leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    reuse=scope.reuse,
                    name=scope.name
                )
                print(conv_result)
            return conv_result

    def __G_loss_fn(self, fake_samples, conds, lambda_):
        """ G loss
        """
        with tf.name_scope('G_loss') as scope:
            # penalize if network no use latent variables
            trainable_V = tf.trainable_variables()
            for v in trainable_V:
                if 'G/latents_linear/weights' in v.name:
                    mean_latents = tf.reduce_mean(tf.abs(v), axis=0)
                    tf.summary.image(
                        'latents_linear_weight', tf.reshape(v, shape=[1, self.latent_dims, 256, 1]), max_outputs=1, collections=['G_weight'])
                if 'G/conds_linear/weights' in v.name:
                    mean_conds = tf.reduce_mean(tf.abs(v), axis=0)
                    tf.summary.image(
                        'conds_linear_weight', tf.reshape(v, shape=[1, 13, 256, 1]), max_outputs=1, collections=['G_weight'])
            penalty_latents_w = lambda_ * tf.reduce_sum(
                tf.abs(mean_latents - mean_conds))
            loss = - \
                tf.reduce_mean(self.critic(fake_samples, conds,
                                           reuse=True)) + penalty_latents_w
        return loss, penalty_latents_w

    def step(self, sess, latent_inputs, conditions):
        """ train one batch on G
        """
        feed_dict = {self.__z: latent_inputs, self.__cond: conditions}
        loss, global_steps, _ = sess.run(
            [self.__G_loss, self.__global_steps,
                self.__G_train_op], feed_dict=feed_dict)
        if self.__G_steps % 1000 == 0:
            summary = sess.run(self.__summary_G_weight_op, feed_dict=feed_dict)
            self.G_summary_writer.add_summary(
                summary, global_step=global_steps)
        if not self.if_log_histogram or self.__G_steps % 100 == 0:  # % 100 to save space
            summary = sess.run(self.__summary_G_op, feed_dict=feed_dict)
            # log
            self.G_summary_writer.add_summary(
                summary, global_step=global_steps)

        self.__G_steps += 1
        return loss, global_steps

    def generate(self, sess, latent_inputs, conditions):
        """ to generate result
        """
        feed_dict = {self.__z: latent_inputs, self.__cond: conditions}
        result = sess.run(self.__G_sample, feed_dict=feed_dict)
        return result
