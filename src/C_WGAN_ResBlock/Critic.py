""" Model of Critic Network
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
    """ Model of Critic Network
    """

    def __init__(self, config, graph, if_training=True):
        """ Build up the graph
        Inputs
        ------
        config :
            * batch_size : mini batch size
            * log_dir : path to save training summary
            * learning_rate : adam's learning rate
            * seq_length : length of sequence during training
            * penalty_lambda = gradient penalty's weight, ref from paper 'improved-wgan'
            * n_resblock : number of resblock in network body
            * if_handcraft_features : if_handcraft_features
            * residual_alpha : residual block = F(x) * residual_alpha + x
            * leaky_relu_alpha : tf.maximum(x, leaky_relu_alpha * x)
            * openshot_penalty_lambda : Critic = Critic - openshot_penalty_lambda * open_shot_score
            * if_use_mismatched : if True, negative scores = mean of (fake_scores + mismatched_scores)
            * n_filters : number of filters in all ConV
        graph :
            tensorflow default graph
        """
        self.data_factory = DataFactory()
        # hyper-parameters
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.seq_length = config.seq_length
        self.penalty_lambda = config.penalty_lambda
        self.n_resblock = config.n_resblock
        self.if_handcraft_features = config.if_handcraft_features
        self.residual_alpha = config.residual_alpha
        self.leaky_relu_alpha = config.leaky_relu_alpha
        self.openshot_penalty_lambda = config.openshot_penalty_lambda
        self.if_use_mismatched = config.if_use_mismatched
        self.n_filters = config.n_filters
        self.if_training = if_training

        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        with tf.name_scope('Critic'):
            self.__steps = tf.get_variable('C_steps', shape=[
            ], dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)
            # data
            self.__G_samples = tf.placeholder(dtype=tf.float32, shape=[
                None, None, 10], name='G_samples')
            self.__real_data = tf.placeholder(dtype=tf.float32, shape=[
                None, None, 10], name='real_data')
            self.__matched_cond = tf.placeholder(dtype=tf.float32, shape=[
                None, None, 13], name='matched_cond')
            self.__mismatched_cond = tf.random_shuffle(self.__matched_cond)
            # adversarial learning : wgan
            self.__build_model()

            # summary
            if self.if_training:
                self.__summary_op = tf.summary.merge(tf.get_collection('C'))
                self.__summary_histogram_op = tf.summary.merge(
                    tf.get_collection('C_histogram'))
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
        self.real_scores = self.inference(
            self.__real_data, self.__matched_cond)
        self.fake_scores = self.inference(
            self.__G_samples, self.__matched_cond, reuse=True)
        if self.if_use_mismatched:
            mismatched_scores = self.inference(
                self.__real_data, self.__mismatched_cond, reuse=True)
            neg_scores = (self.fake_scores + mismatched_scores) / 2.0
        else:
            neg_scores = self.fake_scores

        if self.if_training:
            # loss function
            self.__loss = self.__loss_fn(
                self.__real_data, self.__G_samples, neg_scores, self.real_scores, self.penalty_lambda)
            theta = libs.get_var_list('C')
            with tf.name_scope('optimizer') as scope:
                # Critic train one iteration, step++
                assign_add_ = tf.assign_add(self.__steps, 1)
                with tf.control_dependencies([assign_add_]):
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                    grads = tf.gradients(self.__loss, theta)
                    grads = list(zip(grads, theta))
                    self.__train_op = optimizer.apply_gradients(
                        grads_and_vars=grads, global_step=self.__global_steps)
            # histogram logging
            for grad, var in grads:
                tf.summary.histogram(
                    var.name + '_gradient', grad, collections=['C_histogram'])
        else:
            f_fake = tf.reduce_mean(self.fake_scores)
            f_real = tf.reduce_mean(self.real_scores)
            with tf.name_scope('C_loss') as scope:
                self.EM_dist = f_real - f_fake
                self.summary_em = tf.summary.scalar(
                    'Earth Moving Distance', self.EM_dist)

    def inference(self, inputs, conds, reuse=False):
        """
        Inputs
        ------
        inputs : tensor, float, shape=[batch_size, seq_length=100, features=10]
            real(from data) or fake(from G)
        conds : tensor, float, shape=[batch_size, swq_length=100, features=13]
            conditions, ball and team A
        reuse : bool, optional, defalt value is False
            if share variable

        Return
        ------
        score : float
            real(from data) or fake(from G)
        """
        with tf.variable_scope('C_inference', reuse=reuse):
            concat_ = tf.concat([conds, inputs], axis=-1)
            if self.if_handcraft_features:
                concat_ = self.data_factory.extract_features(concat_)
            with tf.variable_scope('conv_input') as scope:
                conv_input = tf.layers.conv1d(
                    inputs=concat_,
                    filters=self.n_filters,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=libs.leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )
            # residual block
            next_input = conv_input
            for i in range(self.n_resblock):
                res_block = libs.residual_block(
                    'Res' + str(i), next_input, n_filters=self.n_filters, n_layers=2, residual_alpha=self.residual_alpha, leaky_relu_alpha=self.leaky_relu_alpha)
                next_input = res_block
            with tf.variable_scope('conv_output') as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = libs.leaky_relu(normed)
                conv_output = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=1,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    activation=libs.leaky_relu,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer()
                )
                conv_output = tf.reduce_mean(conv_output, axis=1)
                final_ = tf.reshape(
                    conv_output, shape=[-1])
            return final_

    def loss_for_G(self, reals, fakes, conds, latent_weight_penalty):
        """ 
        Param
        -----
        reals : 
        fakes : 
        conds : 
        latent_weight_penalty : 
        """
        openshot_penalty_lambda = tf.constant(
            self.openshot_penalty_lambda)
        openshot_penalty = self.__open_shot_penalty(
            reals, conds, fakes, if_log=True)
        fake_scores = self.inference(fakes, conds, reuse=True)
        scale_ = tf.abs(tf.reduce_mean(fake_scores))
        loss = - tf.reduce_mean(fake_scores) + scale_ * \
            openshot_penalty_lambda * openshot_penalty + scale_ * latent_weight_penalty
        return loss

    def __open_shot_penalty(self, reals, conds, fakes, if_log):
        """
        """
        real_os_penalty = self.__open_shot_score(
            reals, conds, if_log=if_log, log_scope_name='real')
        fake_os_penalty = self.__open_shot_score(
            fakes, conds, if_log=if_log, log_scope_name='fake')
        return tf.abs(real_os_penalty - fake_os_penalty)

    def __open_shot_score(self, inputs, conds, if_log, log_scope_name=''):
        """
        log_scope_name : string
            scope name for open_shot_score
        """
        with tf.name_scope('open_shot_score') as scope:
            # calculate the open shot penalty on each frames
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
            # open shot penalty = amin((theta + 1.0) * (dist_ball_2_teamB + 1.0))
            vec_ball_2_teamB = ball_pos - teamB_pos
            vec_ball_2_basket = ball_pos - basket_pos
            b2teamB_dot_b2basket = tf.matmul(
                vec_ball_2_teamB, vec_ball_2_basket, transpose_b=True)
            b2teamB_dot_b2basket = tf.reshape(b2teamB_dot_b2basket, shape=[
                self.batch_size, self.seq_length, 5])
            dist_ball_2_teamB = tf.norm(
                vec_ball_2_teamB, ord='euclidean', axis=-1)
            dist_ball_2_basket = tf.norm(
                vec_ball_2_basket, ord='euclidean', axis=-1)

            theta = tf.acos(b2teamB_dot_b2basket /
                            (dist_ball_2_teamB * dist_ball_2_basket+1e-3))
            open_shot_score_all = (theta + 1.0) * (dist_ball_2_teamB + 1.0)

            # add
            # one_sub_cosine = 1 - b2teamB_dot_b2basket / \
            #     (dist_ball_2_teamB * dist_ball_2_basket)
            # open_shot_score_all = one_sub_cosine + dist_ball_2_teamB

            open_shot_score_min = tf.reduce_min(
                open_shot_score_all, axis=-1)
            open_shot_score = tf.reduce_mean(open_shot_score_min)

            # too close penalty
            too_close_penalty = 0.0
            for i in range(5):
                vec = tf.subtract(teamB_pos[:, :, i:i+1], teamB_pos)
                dist = tf.sqrt((vec[:, :, :, 0]+1e-8)**2 + (vec[:, :, :, 1]+1e-8)**2)
                too_close_penalty -= tf.reduce_mean(dist)

        if if_log:
            with tf.name_scope(log_scope_name):
                tf.summary.scalar('open_shot_score',
                                  open_shot_score, collections=['G'])
                tf.summary.scalar('too_close_penalty',
                                  too_close_penalty, collections=['G'])
        return open_shot_score + too_close_penalty

    def __loss_fn(self, real_data, G_sample, fake_scores, real_scores, penalty_lambda):
        """ Critic loss

        Params
        ------
        real_data : tensor, float, shape=[batch_size, seq_length, features=10]
            real data, team B, defensive players
        G_sample : tensor, float, shape=[batch_size, seq_length, features=10]
            fake data, team B, defensive players
        fake_scores : tensor, float, shape=[batch_size]
            result from inference given fake data
        real_scores : tensor, float, shape=[batch_size]
            result from inference given real data
        penalty_lambda : float
            gradient penalty's weight, ref from paper 'improved-wgan'

        Return
        ------
        loss : float, shape=[]
            the mean loss of one batch
        """
        with tf.name_scope('C_loss') as scope:
            # grad_pen, base on paper (Improved-WGAN)
            epsilon = tf.random_uniform(
                [self.batch_size, 1, 1], minval=0.0, maxval=1.0)
            X_inter = epsilon * real_data + (1.0 - epsilon) * G_sample
            if self.if_use_mismatched:
                cond_inter = epsilon * self.__matched_cond + \
                    (1.0 - epsilon) * self.__mismatched_cond
            else:
                cond_inter = self.__matched_cond

            grad = tf.gradients(
                self.inference(X_inter, cond_inter, reuse=True), [X_inter])[0]
            sum_ = tf.reduce_sum(tf.square(grad), axis=[1, 2])
            grad_norm = tf.sqrt(sum_)
            grad_pen = penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            EM_dist = tf.identity(real_scores - fake_scores, name="EM_dist")
            f_fake = tf.reduce_mean(fake_scores)
            f_real = tf.reduce_mean(real_scores)
            # Earth Moving Distance
            loss = f_fake - f_real + grad_pen

            # logging
            tf.summary.scalar('C_loss', loss,
                              collections=['C', 'C_valid'])
            tf.summary.scalar('F_real', f_real, collections=['C'])
            tf.summary.scalar('F_fake', f_fake, collections=['C'])
            tf.summary.scalar('Earth Moving Distance',
                              f_real - f_fake, collections=['C', 'C_valid'])
            tf.summary.scalar('grad_pen', grad_pen, collections=['C'])

        return loss

    def step(self, sess, G_samples, real_data, conditions):
        """ train one batch on C

        Params
        ------
        sess : tensorflow Session
        G_samples : float, shape=[batch_size, seq_length, features=10]
            fake data, team B, defensive players
        real_data : float, shape=[batch_size, seq_length, features=10]
            real data, team B, defensive players
        conditions : float, shape=[batch_size, seq_length, features=13]
            real data, team A, offensive players

        Returns
        -------
        loss : float
            batch mean loss
        global_steps : int
            global steps
        """
        feed_dict = {self.__G_samples: G_samples,
                     self.__matched_cond: conditions,
                     self.__real_data: real_data}
        steps, summary, loss, global_steps, _ = sess.run(
            [self.__steps, self.__summary_op, self.__loss, self.__global_steps, self.__train_op], feed_dict=feed_dict)
        # log
        self.summary_writer.add_summary(
            summary, global_step=global_steps)
        if (steps - 1) % 1000 == 0:
            summary_histogram = sess.run(
                self.__summary_histogram_op, feed_dict=feed_dict)
            self.summary_writer.add_summary(
                summary_histogram, global_step=global_steps)

        return loss, global_steps

    def log_valid_loss(self, sess, G_samples, real_data, conditions):
        """ get one batch validation loss

        Params
        ------
        sess : tensorflow Session
        G_samples : float, shape=[batch_size, seq_length, features=10]
            fake data, team B, defensive players
        real_data : float, shape=[batch_size, seq_length, features=10]
            real data, team B, defensive players
        conditions : float, shape=[batch_size, seq_length, features=13]
            real data, team A, offensive players

        Returns
        -------
        loss : float
            validation batch mean loss
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
