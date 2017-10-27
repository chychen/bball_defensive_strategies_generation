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
from utils import Norm


class RNN_WGAN(object):
    """ LSTM + W-GAN
    # TODO list
        * to decide latent variable feature dimentions, 11 for now
        * data precision of float32/  float64
        * num of hidden units
        * dynamic lstm (diff length problems)
        * multi gpus
        * batchnorm
        * 一個分類器來分到底event結束了沒？
        * peephole : pros and cons
        * output feed forward : pros and cons
        * gaussian noise than uniform
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
        self.if_handcrafted = config.if_handcrafted
        # steps
        self.__global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.__G_pretrain_steps = 0
        self.__G_steps = 0
        self.__D_steps = 0
        # data
        self.__z = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.num_features], name='first_frame')
        self.__X = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.seq_length + 1, self.num_features], name='real_data')
        self.__X_real = self.__X[:, :self.seq_length, :]
        self.__X_Label = self.__X[:, 1:, :]
        # supervised learning : pre-training
        self.__build_pretrain()
        # adversarial learning : wgan
        self.__build_wgan()

        self.__summary_G_pretrain_op = tf.summary.merge(
            tf.get_collection('G_pretrain'))
        self.__summary_G_pretrain_valid_op = tf.summary.merge(
            tf.get_collection('G_pretrain_valid'))
        self.__summary_G_op = tf.summary.merge(tf.get_collection('G'))
        self.__summary_D_op = tf.summary.merge(tf.get_collection('D'))
        self.__summary_D_valid_op = tf.summary.merge(
            tf.get_collection('D_valid'))

        # summary writer
        self.G_pretrain_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G_pretrain', graph=graph)
        self.G_pretrain_valid_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G_pretrain_valid', graph=graph)
        self.G_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'G')
        self.D_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'D')
        self.D_valid_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'D_valid')

    def __build_pretrain(self):
        with tf.name_scope('G_pretrain'):
            self.__G_pretrain_sample = self.__G(self.__X_real, seq_len=None, if_pretrain=True)
            # loss function TODO
            with tf.name_scope('G_pretrain_loss'):
                abs_diff = tf.losses.absolute_difference(
                    self.__X_Label, self.__G_pretrain_sample, weights=1.0)
                self.__G_pretrain_loss = tf.reduce_mean(abs_diff)
            # optimizer G-pretrain
            theta_G, _ = self.__get_var_list()
            with tf.name_scope('G_pretrain_optimizer') as scope:
                G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, name="Pretrain_Adam")
                G_grads = tf.gradients(self.__G_pretrain_loss, theta_G)
                G_grads = list(zip(G_grads, theta_G))
                self.__G_pretrain_op = G_optimizer.apply_gradients(
                    grads_and_vars=G_grads, global_step=self.__global_steps)
            # logging
            for grad, var in G_grads:
                self.__summarize(var.name, grad, collections='G_pretrain',
                                 postfix='gradient')
            tf.summary.scalar(
                'G_pretrain_loss', self.__G_pretrain_loss, collections=['G_pretrain', 'G_pretrain_valid'])

    def __build_wgan(self):
        with tf.name_scope('WGAN'):
            self.__G_sample = self.__G(self.__z, seq_len=None, reuse=True)
            # feature extraction
            if self.if_handcrafted:
                real_extracted = self.normer.extract_features(self.__X_real)
                fake_extracted = self.normer.extract_features(self.__G_sample)
            else:
                real_extracted = self.__X_real
                fake_extracted = self.__G_sample
            D_real = self.__D(real_extracted, seq_len=None)
            D_fake = self.__D(fake_extracted, seq_len=None, reuse=True)
            # loss function
            self.__G_loss = self.__G_loss_fn(D_fake)
            self.__D_loss, F_real, F_fake, grad_pen = self.__D_loss_fn(
                real_extracted, fake_extracted, D_fake, D_real, self.penalty_lambda)
            theta_G, theta_D = self.__get_var_list()
            with tf.name_scope('G_optimizer') as scope:
                G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                G_grads = tf.gradients(self.__G_loss, theta_G)
                G_grads = list(zip(G_grads, theta_G))
                self.__G_train_op = G_optimizer.apply_gradients(
                    grads_and_vars=G_grads, global_step=self.__global_steps)
            with tf.name_scope('D_optimizer') as scope:
                D_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)
                D_grads = tf.gradients(self.__D_loss, theta_D)
                D_grads = list(zip(D_grads, theta_D))
                self.__D_train_op = D_optimizer.apply_gradients(
                    grads_and_vars=D_grads, global_step=self.__global_steps)
            # logging
            for grad, var in G_grads:
                self.__summarize(var.name, grad, collections='G',
                                 postfix='gradient')
            for grad, var in D_grads:
                self.__summarize(var.name, grad, collections='D',
                                 postfix='gradient')
            tf.summary.scalar('G_loss', self.__G_loss, collections=['G'])

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
        theta_G = []
        theta_D = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('G'):
                theta_G.append(v)
                self.__summarize(v.op.name, v, collections='G',
                                 postfix='Trainable')
            elif v.name.startswith('D'):
                theta_D.append(v)
                self.__summarize(v.op.name, v, collections='D',
                                 postfix='Trainable')
        return theta_G, theta_D

    def __leaky_relu(self, features, alpha=0.7):
        return tf.maximum(features, alpha * features)

    def __lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                            forget_bias=1.0, state_is_tuple=True,
                            # activation=self.__leaky_relu, cell_clip=2,
                            activation=tf.nn.tanh, reuse=tf.get_variable_scope().reuse)

    def __G(self, inputs, seq_len=None, reuse=False, if_pretrain=False):
        """ TODO
        Inputs
        ------
        inputs : float, shape=[batch, length, dims]
            latent variables
        seq_len : 
            temparily not used

        Return
        ------
        result : float, shape=[batch, length, 23+70]
            generative result (script)
        """
        with tf.variable_scope('G', reuse=reuse) as scope:
            # init
            cell = rnn.MultiRNNCell(
                [self.__lstm_cell() for _ in range(self.rnn_layers)])
            state = cell.zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            # as we feed the output as the input to the next, we 'invent' the
            # initial 'output' as generated_point in the begining. TODO
            generated_point = inputs # first frame
            # model
            output_list = []
            first_player_fc = []
            for time_step in range(self.seq_length):
                fc_merge_list = []
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if if_pretrain:
                    input_ = inputs[:, time_step, :]
                else:
                    input_ = generated_point
                # generated_point = generated_point
                # input_ = 
                # concat_values = [input_]
                # concat_values.append(generated_point)
                # input_ = tf.concat(values=concat_values, axis=1)
                with tf.variable_scope('fully_connect_concat') as scope:
                    lstm_input = layers.fully_connected(
                        inputs=input_,
                        num_outputs=self.hidden_size,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                    self.__summarize('lstm_input', lstm_input, collections=[
                        'G'], postfix='Activation')
                with tf.variable_scope('stack_lstm') as scope:
                    cell_out, state = cell(
                        inputs=lstm_input, state=state, scope=scope)
                    self.__summarize('cell_out', cell_out, collections=[
                        'G'], postfix='Activation')
                with tf.variable_scope('position_fc0') as scope:
                    position_fc0 = layers.fully_connected(
                        inputs=cell_out,
                        num_outputs=23*4,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                with tf.variable_scope('position_fc1') as scope:
                    position_fc1 = layers.fully_connected(
                        inputs=position_fc0,
                        num_outputs=23*2,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                with tf.variable_scope('position_fc') as scope:
                    position_fc = layers.fully_connected(
                        inputs=position_fc1,
                        num_outputs=23,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                    self.__summarize('position_fc', position_fc, collections=[
                        'G'], postfix='Activation')
                    fc_merge_list.append(position_fc)
                with tf.variable_scope('player_fc') as scope:
                    player_fc = layers.fully_connected(
                        inputs=cell_out,
                        num_outputs=70,
                        activation_fn=None,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                    if time_step == 0:
                        # only on first frame
                        with tf.name_scope('10_softmax') as scope:
                            for i in range(0, 70, 7):
                                softmax_out = tf.nn.softmax(
                                    player_fc[:, i:i + 7])
                                first_player_fc.append(softmax_out)
                            first_player_fc = tf.concat(
                                first_player_fc, axis=-1)
                            self.__summarize('first_player_fc', first_player_fc, collections=[
                                'G'], postfix='Activation')
                    fc_merge_list.append(first_player_fc)
                generated_point = tf.concat(fc_merge_list, axis=-1)
                output_list.append(generated_point)
            # stack, axis=1 -> [batch, time, feature]
            result = tf.stack(output_list, axis=1)
            print('result', result)
            return result

    def __D(self, inputs, seq_len=None, reuse=False):
        """
        Inputs
        ------
        inputs : float, shape=[batch, length, 272]
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
            inputs = tf.unstack(inputs, num=self.seq_length, axis=1)
            blstm_input = []
            output_list = []
            with tf.variable_scope('fully_connect_input') as scope:
                for time_step in range(self.seq_length):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    fully_connect_input = layers.fully_connected(
                        inputs=inputs[time_step],
                        num_outputs=self.hidden_size,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.constant_initializer(),
                        scope=scope)
                    self.__summarize('fully_connect_input', fully_connect_input, collections=[
                        'D'], postfix='Activation')
                    blstm_input.append(fully_connect_input)
            with tf.variable_scope('stack_bi_lstm') as scope:
                out_blstm_list, _, _ = rnn.stack_bidirectional_rnn(
                    cells_fw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    cells_bw=[self.__lstm_cell()
                              for _ in range(self.rnn_layers)],
                    inputs=blstm_input,
                    dtype=tf.float32,
                    sequence_length=seq_len,
                    scope=scope
                )
            with tf.variable_scope('fully_connect') as scope:
                for i, out_blstm in enumerate(out_blstm_list):
                    self.__summarize('out_blstm', out_blstm, collections=[
                        'D'], postfix='Activation')
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    fconnect = layers.fully_connected(
                        inputs=out_blstm,
                        num_outputs=1,
                        activation_fn=self.__leaky_relu,
                        weights_initializer=layers.xavier_initializer(
                            uniform=False),
                        biases_initializer=tf.zeros_initializer(),
                        scope=scope)
                    self.__summarize('fconnect', fconnect, collections=[
                        'D'], postfix='Activation')
                    output_list.append(fconnect)
            # print(output_list)
            # stack, axis=1 -> [batch, time, feature]
            decisions = tf.stack(output_list, axis=1)
            print('decisions', decisions)
            decision = tf.reduce_mean(decisions, axis=1)
            print('decision', decision)

            return decision

    def __G_loss_fn(self, D_fake):
        """ G loss
        """
        with tf.name_scope('G_loss') as scope:
            loss = -tf.reduce_mean(D_fake)
        return loss

    def __D_loss_fn(self, __X, __G_sample, D_fake, D_real, penalty_lambda):
        """ D loss
        """
        with tf.name_scope('D_loss') as scope:
            # grad_pen, base on paper (Improved WGAN)
            epsilon = tf.random_uniform(
                [self.batch_size, 1, 1], minval=0.0, maxval=1.0)
            __X_inter = epsilon * __X + (1.0 - epsilon) * __G_sample
            grad = tf.gradients(
                self.__D(__X_inter, seq_len=None, reuse=True), [__X_inter])[0]
            sum_ = tf.reduce_sum(tf.square(grad), axis=[1, 2])
            grad_norm = tf.sqrt(sum_)
            grad_pen = penalty_lambda * tf.reduce_mean(
                tf.square(grad_norm - 1.0))
            f_fake = tf.reduce_mean(D_fake)
            f_real = tf.reduce_mean(D_real)

            loss = f_fake - f_real + grad_pen
        return loss, f_real, f_fake, grad_pen

    def G_pretrain_step(self, sess, real_data):
        """ train one batch on G
        """
        self.__G_pretrain_steps += 1
        feed_dict = {self.__X: real_data}
        loss, global_steps, _ = sess.run(
            [self.__G_pretrain_loss, self.__global_steps,
                self.__G_pretrain_op], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__G_pretrain_steps % 100 == 0:  # % 100 to save space
            summary = sess.run(self.__summary_G_pretrain_op,
                               feed_dict=feed_dict)
            # log
            self.G_pretrain_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss, global_steps

    def G_pretrain_log_valid_loss(self, sess, real_data):
        """ train one batch on G
        """
        feed_dict = {self.__X: real_data}
        loss, global_steps = sess.run([self.__G_pretrain_loss, self.__global_steps], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__G_pretrain_steps % 100 == 0:  # % 100 to save space
            summary = sess.run(self.__summary_G_pretrain_valid_op,
                               feed_dict=feed_dict)
            # log
            self.G_pretrain_valid_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss

    def G_step(self, sess, latent_inputs):
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

    def D_step(self, sess, latent_inputs, real_data):
        """ train one batch on D
        """
        self.__D_steps += 1
        feed_dict = {self.__z: latent_inputs,
                     self.__X: real_data}
        loss, global_steps, _ = sess.run(
            [self.__D_loss, self.__global_steps, self.__D_train_op], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__D_steps % 500 == 0:  # % 500 to save space
            summary = sess.run(self.__summary_D_op, feed_dict=feed_dict)
            # log
            self.D_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss, global_steps

    def D_log_valid_loss(self, sess, latent_inputs, real_data):
        """ one batch valid loss
        """
        feed_dict = {self.__z: latent_inputs,
                     self.__X: real_data}
        loss, global_steps = sess.run(
            [self.__D_loss, self.__global_steps], feed_dict=feed_dict)
        if not self.if_log_histogram or self.__D_steps % 500 == 0:  # % 500 to save space
            summary = sess.run(self.__summary_D_valid_op, feed_dict=feed_dict)
            # log
            self.D_valid_summary_writer.add_summary(
                summary, global_step=global_steps)
        return loss

    def generate_pretrain(self, sess, real_data):
        """ to generate result
        """
        feed_dict = {self.__X: real_data}
        result = sess.run(self.__G_pretrain_sample, feed_dict=feed_dict)
        return result

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
