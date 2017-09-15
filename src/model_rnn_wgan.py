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


class RNN_WGAN(object):
    """ LSTM + W-GAN
    # TODO list
        * to decide latent variable feature dimentions, 11 for now
        * data precision of float32/  float64
        * num of hidden units
        * output feedback to next input
        * dynamic lstm (diff length problems)
        * multi gpus
        * leaky relu
        * batchnorm
        * gradient penalty
    """

    def __init__(self, config, graph):
        """ TO build up the computational graph
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
        self.if_feed_previous = config.if_feed_previous
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        # data
        self.__z = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, self.latent_dims], name='latent_input')
        self.__X = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length, self.num_features], name='real_data')
        # model
        self.__G_sample = self.__G(self.__z, seq_len=None)
        D_real = self.__D(self.__X, seq_len=None)
        D_fake = self.__D(self.__G_sample, is_fake=True)
        # loss function
        self.__G_loss = self.__G_loss_fn(D_fake)
        self.__D_loss = self.__D_loss_fn(
            self.__X, self.__G_sample, D_fake, D_real, self.penalty_lambda)
        # optimizer
        theta_G, theta_D = self.__get_var_list()
        self.__G_solver = (tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
                           .minimize(self.__G_loss, var_list=theta_G, global_step=self.__global_steps))
        self.__D_solver = (tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
                           .minimize(self.__D_loss, var_list=theta_D, global_step=self.__global_steps))
        # summary
        # use self.__loss_V to draw both D's and G's loss on same plot
        self.__loss_V = tf.Variable(0.0)
        self.__merged_op = tf.summary.scalar('loss', self.__loss_V)
        # summary writer
        self.G_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G', graph=graph)
        self.D_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'D', graph=graph)

    def __get_var_list(self):
        """ to get both Generator's and Discriminator's trainable variables
        """
        trainable_V = tf.trainable_variables()
        theta_G = []
        theta_D = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('G'):
                theta_G.append(v)
            elif v.name.startswith('D'):
                theta_D.append(v)
        return theta_G, theta_D

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                            forget_bias=1.0, state_is_tuple=True,
                            activation=tf.tanh, reuse=tf.get_variable_scope().reuse)

    def __G(self, inputs, seq_len=None):
        """
        Inputs
        ------
        inputs : float, shape=[batch, length, dims]
            latent variables
        seq_len : 
            temparily not used

        Return
        ------
        result : float, shape=[batch, length, 23]
            generative result (script)
        """
        with tf.variable_scope('G') as scope:
            # init
            cell = rnn.MultiRNNCell(
                [self.__lstm_cell() for _ in range(self.rnn_layers)])
            state = cell.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            # as we feed the output as the input to the next, we 'invent' the
            # initial 'output' as generated_point in the begining.
            generated_point = tf.random_uniform(
                shape=[self.batch_size, self.num_features], minval=0.0, maxval=1.0)
            # model
            output_list = []
            for time_step in range(self.seq_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                input_ = inputs[:, time_step, :]
                concat_values = [input_]
                if self.if_feed_previous:
                    concat_values.append(generated_point)
                input_ = tf.concat(values=concat_values, axis=1)
                with tf.variable_scope('fully_connect_concat') as scope:
                    lstm_input = layers.fully_connected(
                        inputs=input_,
                        num_outputs=self.hidden_size,
                        activation_fn=tf.nn.relu,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                with tf.variable_scope('stack_lstm') as scope:
                    cell_out, state = cell(
                        inputs=lstm_input, state=state, scope=scope)
                with tf.variable_scope('fully_connect_output') as scope:
                    generated_point = layers.fully_connected(
                        inputs=cell_out,
                        num_outputs=self.num_features,
                        activation_fn=tf.nn.relu,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                output_list.append(generated_point)
            # stack, axis=1 -> [batch, time, feature]
            result = tf.stack(output_list, axis=1)
            print('result', result)
            return result

    def __D(self, inputs, seq_len=None, is_fake=False):
        """
        Inputs
        ------
        inputs : float, shape=[batch, length, 23]
            real(from data) or fake(from G)
        seq_len : 
            temparily not used

        Return
        ------
        decision : bool
            real(from data) or fake(from G)
        """
        # unstack, axis=1 -> [batch, time, feature]
        inputs = tf.unstack(inputs, num=self.seq_length, axis=1)
        with tf.variable_scope('D') as scope:
            output_list = []
            if is_fake:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('stack_bi_lstm') as scope:
                out_blstm_list, _, _ = rnn.stack_bidirectional_rnn(
                    cells_fw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    cells_bw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    inputs=inputs,
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
                        activation_fn=tf.nn.relu,
                        weights_initializer=layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                output_list.append(fconnect)
            # stack, axis=1 -> [batch, time, feature]
            decisions = tf.stack(output_list, axis=1)
            print('decisions', decisions)
            decision = tf.reduce_mean(decisions, axis=1)
            print('decision', decision)

            return decision

    def __G_loss_fn(self, D_fake):
        """ G loss
        """
        loss = -tf.reduce_mean(D_fake)
        return loss

    def __D_loss_fn(self, __X, __G_sample, D_fake, D_real, penalty_lambda):
        """ D loss
        """
        # grad_pen, base on paper (Improved WGAN)
        epsilon = tf.random_uniform(
            [self.batch_size, 1, 1], minval=0.0, maxval=1.0)
        __X_inter = epsilon * __X + (1.0 - epsilon) * __G_sample
        grad = tf.gradients(self.__D(__X_inter, is_fake=True), [__X_inter])[0]
        # TODO check:
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = tf.reduce_mean((grad_norm - 1.0)**2)
        # grad_pen = tf.reduce_mean((tf.abs(grad) - 1.0)**2)

        loss = tf.reduce_mean(
            D_fake) - tf.reduce_mean(D_real) + penalty_lambda * grad_pen
        return loss

    def G_step(self, sess, latent_inputs):
        """ train one batch on G
        """
        feed_dict = {self.__z: latent_inputs}
        loss, global_steps, _ = sess.run(
            [self.__G_loss, self.__global_steps,
                self.__G_solver], feed_dict=feed_dict)
        # log
        summary = sess.run(self.__merged_op, feed_dict={self.__loss_V: loss})
        self.G_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss, global_steps

    def D_step(self, sess, latent_inputs, real_data):
        """ train one batch on D
        """
        feed_dict = {self.__z: latent_inputs, self.__X: real_data}
        loss, global_steps, _ = sess.run(
            [self.__D_loss, self.__global_steps,
                self.__D_solver], feed_dict=feed_dict)
        # log
        summary = sess.run(self.__merged_op, feed_dict={self.__loss_V: loss})
        self.D_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss, global_steps

    def generate(self, sess, latent_inputs):
        """ to generate result
        """
        feed_dict = {self.__z: latent_inputs}
        result = sess.run(self.__G_sample, feed_dict=feed_dict)
        return result


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.total_epoches = 100
        self.num_train_D = 5
        self.batch_size = 64
        self.log_dir = 'test_only/'
        self.learning_rate = 1e-3
        self.hidden_size = 110
        self.rnn_layers = 1
        self.seq_length = 5
        self.num_features = 23
        self.latent_dims = 11
        self.penalty_lambda = 10


def test():
    """ testing only
    """
    with tf.Graph().as_default() as g:
        config = TestingConfig()
        model = RNN_WGAN(config, graph=g)
        # dummy test on training
        Z = np.zeros(shape=[config.batch_size,
                            config.seq_length, config.latent_dims])
        X = np.zeros(shape=[config.batch_size,
                            config.seq_length, config.num_features])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(config.total_epoches):
                for i in range(config.num_train_D):
                    D_loss, global_steps = model.D_step(sess, Z, X)
                    print('D_loss', D_loss)
                    print('global_steps', global_steps)
                G_loss, global_steps = model.G_step(sess, Z)
                print('G_loss', G_loss)
                print('global_steps', global_steps)


if __name__ == '__main__':
    if os.path.exists('test_only'):
        shutil.rmtree('test_only')
        print('rm -rf "test_only" complete!')
    test()
