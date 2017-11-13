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
            * seq_length : length of LSTM
            * latent_dims : dimensions of latent feature
            * penalty_lambda = gradient penalty's weight, ref from  paper of 'improved-wgan'
        graph : 
            tensorflow default graph
        """
        self.data_factory = DataFactory()
        # hyper-parameters
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.seq_length = config.seq_length
        self.latent_dims = config.latent_dims
        self.penalty_lambda = config.penalty_lambda
        self.latent_penalty_lambda = config.latent_penalty_lambda
        self.n_resblock = config.n_resblock
        self.if_feed_extra_info = config.if_feed_extra_info
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.__steps = 0
        # IO
        self.critic = critic_inference
        self.__z = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.latent_dims], name='latent_input')
        self.__cond = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, 13], name='team_a')
        # adversarial learning : wgan
        self.__build_model()

        # summary
        self.__summary_op = tf.summary.merge(tf.get_collection('G'))
        self.__summary_histogram_op = tf.summary.merge(
            tf.get_collection('G_histogram'))
        self.__summary_weight_op = tf.summary.merge(
            tf.get_collection('G_weight'))
        self.summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G', graph=graph)

    def __build_model(self):
        with tf.name_scope('Generator'):
            self.__G_sample = self.__G(self.__z, self.__cond)
            # loss function
            self.__loss, self.__penalty_latents_w, self.__critic_scores = self.__G_loss_fn(
                self.__G_sample, self.__cond, lambda_=self.latent_penalty_lambda)
            theta = libs.get_var_list('G')
            with tf.name_scope('optimizer') as scope:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                grads = tf.gradients(self.__loss, theta)
                grads = list(zip(grads, theta))
                self.__train_op = optimizer.apply_gradients(
                    grads_and_vars=grads, global_step=self.__global_steps)
            for grad, var in grads:
                tf.summary.histogram(
                    var.name + '_gradient', grad, collections=['G_histogram'])
            # logging
            tf.summary.scalar('G_loss', self.__loss, collections=['G'])
            tf.summary.scalar('G_penalty_latents_w',
                              self.__penalty_latents_w, collections=['G'])

    def __G(self, latents, conds):
        """
        Inputs
        ------
        latents : float, shape=[batch, dims]
            latent variables
        conds : float, shape=[batch, length=100, dims=13]
            conditional constraints of team A and ball

        Return
        ------
        result : float, shape=[batch, length=100, 10]
            generative result (team b)
        """
        with tf.variable_scope('G_inference'):  # init
            input_ = conds
            if self.if_feed_extra_info:
                with tf.name_scope('concat_info'):
                    left_x = tf.constant(self.data_factory.BASKET_LEFT[0], dtype=tf.float32, shape=[
                        self.batch_size, self.seq_length, 1])
                    left_y = tf.constant(self.data_factory.BASKET_LEFT[1], dtype=tf.float32, shape=[
                        self.batch_size, self.seq_length, 1])
                    right_x = tf.constant(self.data_factory.BASKET_RIGHT[0], dtype=tf.float32, shape=[
                        self.batch_size, self.seq_length, 1])
                    right_y = tf.constant(self.data_factory.BASKET_RIGHT[1], dtype=tf.float32, shape=[
                        self.batch_size, self.seq_length, 1])
                    input_ = tf.concat(
                        [left_x, left_y, right_x, right_y, conds], axis=-1)
            with tf.variable_scope('conds_linear') as scope:
                conds_linear = layers.fully_connected(
                    inputs=input_,
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
            # print(next_input)
            # residual block
            for i in range(self.n_resblock):
                res_block = libs.residual_block('Res' + str(i), next_input)
                next_input = res_block
                # print(next_input)
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
                    bias_initializer=tf.zeros_initializer()
                )
                # print(conv_result)
            return conv_result

    def __G_loss_fn(self, fake_samples, conds, lambda_):
        """ G loss
        """
        with tf.name_scope('G_loss') as scope:
            # penalize if network no use latent variables
            trainable_V = tf.trainable_variables()
            for v in trainable_V:
                if 'G_inference/latents_linear/weights' in v.name:
                    mean_latents = tf.reduce_mean(tf.abs(v), axis=0)
                    shape_ = v.get_shape().as_list()
                    tf.summary.image(
                        'latents_linear_weight', tf.reshape(v, shape=[1, shape_[0], shape_[1], 1]), max_outputs=1, collections=['G_weight'])
                if 'G_inference/conds_linear/weights' in v.name:
                    mean_conds = tf.reduce_mean(tf.abs(v), axis=0)
                    shape_ = v.get_shape().as_list()
                    tf.summary.image(
                        'conds_linear_weight', tf.reshape(v, shape=[1, shape_[0], shape_[1], 1]), max_outputs=1, collections=['G_weight'])
            penalty_latents_w = tf.reduce_mean(
                tf.abs(mean_latents - mean_conds))
            critic_scores = self.critic(fake_samples, conds, reuse=True)
            loss = - \
                tf.reduce_mean(critic_scores) + lambda_ * penalty_latents_w
        return loss, penalty_latents_w, critic_scores

    def step(self, sess, latent_inputs, conditions):
        """ train one batch on G
        """
        feed_dict = {self.__z: latent_inputs, self.__cond: conditions}
        summary, loss, global_steps, _ = sess.run(
            [self.__summary_op, self.__loss, self.__global_steps,
                self.__train_op], feed_dict=feed_dict)
        # log
        self.summary_writer.add_summary(
            summary, global_step=global_steps)
        if self.__steps % 200 == 0:
            summary, summary_trainable = sess.run(
                [self.__summary_weight_op, self.__summary_histogram_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(
                summary, global_step=global_steps)
            self.summary_writer.add_summary(
                summary_trainable, global_step=global_steps)

        self.__steps += 1
        return loss, global_steps

    def generate(self, sess, latent_inputs, conditions):
        """ to generate result

        Returns
        -------
        result : float32, shape=[batch_size, 10]
            positions b team's players 
        """
        feed_dict = {self.__z: latent_inputs, self.__cond: conditions}
        result = sess.run(self.__G_sample, feed_dict=feed_dict)
        return result
