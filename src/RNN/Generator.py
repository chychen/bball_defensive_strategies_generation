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

        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.num_features = config.num_features
        self.if_feed_prev = True

        self.penalty_lambda = config.penalty_lambda
        self.latent_penalty_lambda = config.latent_penalty_lambda
        self.n_resblock = config.n_resblock
        self.if_feed_extra_info = config.if_feed_extra_info
        self.residual_alpha = config.residual_alpha
        self.leaky_relu_alpha = config.leaky_relu_alpha
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
        #self.__summary_weight_op = tf.summary.merge(
            #tf.get_collection('G_weight'))
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

    def __leaky_relu(self, features):
        return tf.maximum(features, self.leaky_relu_alpha * features)

    def _lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size,use_peepholes=True,initializer=None,
                            forget_bias=1.0,state_is_tuple=True,activation=tf.tanh,
                            reuse=tf.get_variable_scope().reuse)

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
        with tf.variable_scope('G'):
            '''
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
            '''
            cell = rnn.MultiRNNCell([self._lstm_cell() for _ in range(self.num_layers)])
            state = cell.zero_state(batch_size=self.batch_size,dtype=tf.float32)
            generated_point = latents
            inputs = conds
            output_ = []
            for time_step in range(self.seq_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                input_ = inputs[:,time_step,:]
                concat_val = [input_]
                concat_val.append(generated_point)
                input_ = tf.concat(values = concat_val,axis = 1)
                with tf.variable_scope('fully_connect') as scope:
                    lstm_input = layers.fully_connected(inputs=input_,
                                                        num_outputs=self.hidden_size,
                                                        activation_fn=self.__leaky_relu,
                                                        weights_initializer=layers.xavier_initializer(uniform=False),
                                                        biases_initializer=tf.zeros_initializer(),
                                                        scope=scope)


                with tf.variable_scope('stacked_lstm') as scope:

                    cell_out, state = cell(inputs=lstm_input,state=state,scope=scope)


                with tf.variable_scope('position_fc') as scope:
                    position_fc = layers.fully_connected(inputs=cell_out,
                                                         num_outputs=10,
                                                         activation_fn=self.__leaky_relu,
                                                         weights_initializer=layers.xavier_initializer(uniform=False),
                                                         biases_initializer=tf.zeros_initializer(),
                                                         scope=scope)
                generated_point = position_fc
                output_.append(generated_point)
            result = tf.stack(output_,axis=1)
            return result

    def __G_loss_fn(self, fake_samples, conds, lambda_):
        """ G loss
        """
        with tf.name_scope('G_loss') as scope:
            # penalize if network no use latent variables
            mean_latents = 1.0
            mean_conds = 1.0
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
                tf.reduce_mean(critic_scores) + lambda_
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
       # if self.__steps % 200 == 0:
            #summary, summary_trainable = sess.run(
                #[self.__summary_histogram_op], feed_dict=feed_dict)
            #self.summary_writer.add_summary(
                #summary, global_step=global_steps)
            #self.summary_writer.add_summary(
                #summary_trainable, global_step=global_steps)

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
