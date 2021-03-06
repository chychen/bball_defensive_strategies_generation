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


class C_MODEL(object):
    """ 
    """

    def __init__(self, config, graph, if_training=True):
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
        # RNN parameters
        self.num_features = config.num_features
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.if_training = if_training

        self.penalty_lambda = config.penalty_lambda
        self.if_handcraft_features = config.if_handcraft_features
        self.leaky_relu_alpha = config.leaky_relu_alpha
        self.heuristic_penalty_lambda = config.heuristic_penalty_lambda
        self.if_use_mismatched = config.if_use_mismatched
        self.if_trainable_lambda = config.if_trainable_lambda
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.__steps = 0
        # data
        self.__G_samples = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, 10], name='G_samples')
        self.__real_data = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, 10], name='real_data')
        self.__matched_cond = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, 13], name='matched_cond')
        self.__mismatched_cond = tf.random_shuffle(self.__matched_cond)
        # adversarial learning : wgan
        self.__build_model()

        # summary
        if self.if_training:
            self.__summary_op = tf.summary.merge(tf.get_collection('C'))
            # self.__summary_histogram_op = tf.summary.merge(
            # tf.get_collection('C_histogram'))
            self.__summary_valid_op = tf.summary.merge(
                tf.get_collection('C_valid'))
            self.summary_writer = tf.summary.FileWriter(
                self.log_dir + 'C')
            self.valid_summary_writer = tf.summary.FileWriter(
                self.log_dir + 'C_valid')
        else:
            self.baseline_summary_writer = tf.summary.FileWriter(
                self.log_dir + 'Baseline_C')

    def __build_model(self):
        with tf.name_scope('Critic'):
            # inference
            real_scores = self.inference(
                self.__real_data, self.__matched_cond, seq_len=None, if_log_scalar_summary=True, log_scope_name='real_scores')
            fake_scores = self.inference(
                self.__G_samples, self.__matched_cond, reuse=True, if_log_scalar_summary=True, log_scope_name='fake_scores')
            # mismatched_scores = self.inference(
            # self.__real_data, self.__mismatched_cond, reuse=True)
            # if self.if_use_mismatched:
            #neg_scores = (fake_scores + mismatched_scores) / 2.0
            # else:
            neg_scores = fake_scores
            if self.if_training:
                # loss function
                self.__loss = self.__loss_fn(
                    self.__real_data, self.__G_samples, neg_scores, real_scores, self.penalty_lambda)

                theta = libs.get_var_list('C')

                with tf.name_scope('optimizer') as scope:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate, beta1=0.5)
                    grads = tf.gradients(self.__loss, theta)
                    grads = list(zip(grads, theta))
                    self.__train_op = optimizer.apply_gradients(
                        grads_and_vars=grads, global_step=self.__global_steps)
                # for grad, var in grads:
                    # tf.summary.histogram(
                    # var.name + '_gradient', grad, collections=['C_histogram'])
            else:
                f_fake = tf.reduce_mean(fake_scores)
                f_real = tf.reduce_mean(real_scores)
                with tf.name_scope('C_loss') as scope:
                    self.EM_dist = f_real - f_fake
                    self.summary_em = tf.summary.scalar(
                        'Earth Moving Distance', self.EM_dist)

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, initializer=None,
                            forget_bias=1.0, state_is_tuple=True)

    def inference(self, inputs, conds, seq_len=None, reuse=False, if_log_scalar_summary=False, log_scope_name=''):
        """
        Inputs
        ------
        inputs : float, shape=[batch_size, seq_length=100, features=10]
            real(from data) or fake(from G)
        conds : float, shape=[batch_size, swq_length=100, features=13]

        Return
        ------
        score : float
            real(from data) or fake(from G)
        """
        concat_ = tf.concat([conds, inputs], axis=-1)
        inputs_ = tf.unstack(concat_, num=self.seq_length, axis=1)
        with tf.variable_scope('C_inference') as scope:
            output_list = []
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('stack_bi_lstm') as scope:
                out_blstm_list, _, _ = rnn.stack_bidirectional_rnn(
                    cells_fw=[self.__lstm_cell()
                              for _ in range(self.num_layers)],
                    cells_bw=[self.__lstm_cell()
                              for _ in range(self.num_layers)],
                    inputs=inputs_,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope=scope
                )
            for i, out_blstm in enumerate(out_blstm_list):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('fully_connect') as scope:
                    fconnect = layers.fully_connected(
                        inputs=out_blstm,
                        num_outputs=1,
                        activation_fn=libs.leaky_relu,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                output_list.append(fconnect)
            # stack, axis=1 -> [batch, time, feature]
            decisions = tf.stack(output_list, axis=1)
            final_ = tf.reduce_mean(decisions, axis=1)
            final_ = tf.reshape(final_, shape=[self.batch_size])

            with tf.name_scope('heuristic_penalty') as scope:
                # 0. prepare data
                ball_pos = tf.reshape(conds[:, :, :2], shape=[
                                      self.batch_size, self.seq_length, 1, 2])
                teamB_pos = tf.reshape(
                    inputs, shape=[self.batch_size, self.seq_length, 5, 2])
                basket_right_x = tf.constant(self.data_factory.BASKET_RIGHT[0], dtype=tf.float32, shape=[
                    self.batch_size, self.seq_length, 1, 1])
                basket_right_y = tf.constant(self.data_factory.BASKET_RIGHT[1], dtype=tf.float32, shape=[
                    self.batch_size, self.seq_length, 1, 1])
                basket_pos = tf.concat(
                    [basket_right_x, basket_right_y], axis=-1)

                vec_ball_2_teamB = ball_pos - teamB_pos  # [128,100,5,2]
                vec_ball_2_basket = ball_pos - basket_pos  # [128,100,1,2]
                b2teamB_dot_b2basket = tf.matmul(
                    vec_ball_2_teamB, vec_ball_2_basket, transpose_b=True)  # [128,100,5,1]
                b2teamB_dot_b2basket = tf.reshape(b2teamB_dot_b2basket, shape=[
                    self.batch_size, self.seq_length, 5])
                dist_ball_2_teamB = tf.norm(
                    vec_ball_2_teamB, ord='euclidean', axis=-1)
                dist_ball_2_basket = tf.norm(
                    vec_ball_2_basket, ord='euclidean', axis=-1)
                one_sub_cosine = 1 - b2teamB_dot_b2basket / \
                    (dist_ball_2_teamB * dist_ball_2_basket)
                heuristic_penalty_all = one_sub_cosine * dist_ball_2_teamB
                heuristic_penalty_min = tf.reduce_min(
                    heuristic_penalty_all, axis=-1)
                heuristic_penalty = tf.reduce_mean(heuristic_penalty_min)

                if self.if_trainable_lambda:
                    trainable_lambda = tf.get_variable('trainable_heuristic_penalty_lambda', shape=[
                    ], dtype=tf.float32, initializer=tf.constant_initializer(value=1.0))
                else:
                    trainable_lambda = tf.constant(
                        self.heuristic_penalty_lambda)

            # logging
            if if_log_scalar_summary:
                with tf.name_scope(log_scope_name):
                    tf.summary.scalar('heuristic_penalty',
                                      heuristic_penalty, collections=['C'])
                    tf.summary.scalar('trainable_lambda',
                                      trainable_lambda, collections=['C'])

            return final_ - trainable_lambda * heuristic_penalty

    def __loss_fn(self, real_data, G_sample, fake_scores, real_scores, penalty_lambda):
        """ C loss
        """
        with tf.name_scope('C_loss') as scope:
            # grad_pen, base on paper (Improved WGAN)
            epsilon = tf.random_uniform(
                [self.batch_size, 1, 1], minval=0.0, maxval=1.0)
            X_inter = epsilon * real_data + (1.0 - epsilon) * G_sample

            grad = tf.gradients(
                self.inference(X_inter, self.__matched_cond, reuse=True), [X_inter])[0]
            print(grad)
            sum_ = tf.reduce_sum(tf.square(grad), axis=[1, 2])
            # print(sum_)
            grad_norm = tf.sqrt(sum_)
            grad_pen = penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            f_fake = tf.reduce_mean(fake_scores)
            f_real = tf.reduce_mean(real_scores)

            loss = f_fake - f_real + grad_pen

            # logging
            tf.summary.scalar('C_loss', loss,
                              collections=['C', 'C_valid'])
            tf.summary.scalar('F_real', f_real, collections=['C'])
            tf.summary.scalar('F_fake', f_fake, collections=['C'])
            tf.summary.scalar('Earth Moving Distance', f_real -
                              f_fake, collections=['C', 'C_valid'])
            tf.summary.scalar('grad_pen', grad_pen, collections=['C'])

        return loss

    def step(self, sess, G_samples, real_data, conditions):
        """ train one batch on C
        """
        feed_dict = {self.__G_samples: G_samples,
                     self.__matched_cond: conditions,
                     self.__real_data: real_data}
        summary, loss, global_steps, _ = sess.run(
            [self.__summary_op, self.__loss, self.__global_steps, self.__train_op], feed_dict=feed_dict)
        # log
        self.summary_writer.add_summary(
            summary, global_step=global_steps)
        # if self.__steps % 1000 == 0:
        # summary_histogram = sess.run(
        # self.__summary_histogram_op, feed_dict=feed_dict)
        # self.summary_writer.add_summary(
        # summary_histogram, global_step=global_steps)

        self.__steps += 1
        return loss, global_steps

    def log_valid_loss(self, sess, G_samples, real_data, conditions):
        """ one batch valid loss
        """
        feed_dict = {self.__G_samples: G_samples,
                     self.__matched_cond: conditions,
                     self.__real_data: real_data}
        summary, loss, global_steps = sess.run(
            [self.__summary_valid_op, self.__loss, self.__global_steps], feed_dict=feed_dict)
        # log
        self.valid_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss

    def eval_EM_distance(self, sess, G_samples, real_data, conditions, global_steps):
        """ 
        """
        feed_dict = {self.__G_samples: G_samples,
                     self.__matched_cond: conditions,
                     self.__real_data: real_data}
        _, summary = sess.run(
            [self.EM_dist, self.summary_em], feed_dict=feed_dict)
        self.baseline_summary_writer.add_summary(
            summary, global_step=global_steps)
