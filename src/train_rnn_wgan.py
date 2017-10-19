"""
data url: http://140.113.210.14:6006/NBA/data/NBA-TEAM1.npy
data description: 
    event by envet, with 300 sequence for each. (about 75 seconds)
    shape as [number of events, max sequence length, 33 dimensions(1 ball and 10 players x,y,z)]
    save it under the relative path './data/' before training
    # TODO visulize the player positions on gif file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import model_rnn_wgan
import game_visualizer
from utils import Norm

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string('log_dir', 'v11/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v11/checkpoints/',
                           "checkpoints dir")
tf.app.flags.DEFINE_string('sample_dir', 'v11/sample/',
                           "directory to save generative result")
tf.app.flags.DEFINE_string('data_path', '../data/FEATURES.npy',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# input parameters
tf.app.flags.DEFINE_integer('seq_length', 100,
                            "the maximum length of one training data")
tf.app.flags.DEFINE_integer('num_features', 23 + 70,
                            "3 (ball x y z) + 10 (players) * 2 (x and y) + 70 (player positions as 10 7-dims-one-hot)")
tf.app.flags.DEFINE_integer('latent_dims', 10,
                            "dimensions of latant variable")
# training parameters
tf.app.flags.DEFINE_integer('total_epoches', 3000,
                            "num of ephoches")
tf.app.flags.DEFINE_integer('num_train_D', 5,
                            "num of times of training D before train G")
tf.app.flags.DEFINE_integer('num_pretrain_D', 10,
                            "num of ephoch to train D before train G")
tf.app.flags.DEFINE_integer('freq_train_D', 50,
                            "freqence of num ephoch to train D more")
tf.app.flags.DEFINE_integer('batch_size', 64,
                            "batch size")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          "learning rate")
tf.app.flags.DEFINE_integer('hidden_size', 230,
                            "hidden size of LSTM")
tf.app.flags.DEFINE_integer('rnn_layers', 2,
                            "num of layers for rnn")
tf.app.flags.DEFINE_float('penalty_lambda', 10.0,
                          "regularization parameter of wGAN loss function")
tf.app.flags.DEFINE_bool('if_feed_previous', True,
                         "if feed the previous output concated with current input")
tf.app.flags.DEFINE_integer('pretrain_epoches', 0,
                            "num of ephoch to train label as input")
# logging
tf.app.flags.DEFINE_integer('save_model_freq', 50,
                            "num of epoches to save model")
tf.app.flags.DEFINE_integer('save_result_freq', 20,
                            "num of epoches to save gif")
tf.app.flags.DEFINE_integer('log_freq', 200,
                            "num of steps to log")


class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self):
        self.total_epoches = FLAGS.total_epoches
        self.batch_size = FLAGS.batch_size
        self.log_dir = FLAGS.log_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.sample_dir = FLAGS.sample_dir
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.hidden_size = FLAGS.hidden_size
        self.rnn_layers = FLAGS.rnn_layers
        self.save_model_freq = FLAGS.save_model_freq
        self.save_result_freq = FLAGS.save_result_freq
        self.log_freq = FLAGS.log_freq
        self.seq_length = FLAGS.seq_length
        self.num_features = FLAGS.num_features
        self.latent_dims = FLAGS.latent_dims
        self.penalty_lambda = FLAGS.penalty_lambda
        self.if_feed_previous = FLAGS.if_feed_previous
        self.num_train_D = FLAGS.num_train_D
        self.num_pretrain_D = FLAGS.num_pretrain_D
        self.freq_train_D = FLAGS.freq_train_D
        self.pretrain_epoches = FLAGS.pretrain_epoches

    def show(self):
        print("total_epoches:", self.total_epoches)
        print("batch_size:", self.batch_size)
        print("log_dir:", self.log_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("sample_dir:", self.sample_dir)
        print("data_path:", self.data_path)
        print("learning_rate:", self.learning_rate)
        print("hidden_size:", self.hidden_size)
        print("rnn_layers:", self.rnn_layers)
        print("save_model_freq:", self.save_model_freq)
        print("save_result_freq:", self.save_result_freq)
        print("log_freq:", self.log_freq)
        print("seq_length:", self.seq_length)
        print("num_features:", self.num_features)
        print("latent_dims:", self.latent_dims)
        print("penalty_lambda:", self.penalty_lambda)
        print("if_feed_previous:", self.if_feed_previous)
        print("num_train_D:", self.num_train_D)
        print("num_pretrain_D:", self.num_pretrain_D)
        print("freq_train_D:", self.freq_train_D)
        print("pretrain_epoches:", self.pretrain_epoches)


def z_samples():
    # TODO sample z from normal-distribution than
    return np.random.uniform(
        -1., 1., size=[FLAGS.batch_size, FLAGS.seq_length, FLAGS.latent_dims])


def training(sess, model, real_data, num_batches, saver, normer, is_pretrain=False):
    """
    """

    shuffled_indexes = np.random.permutation(real_data.shape[0])
    real_data = real_data[shuffled_indexes]
    real_data, valid_data = np.split(real_data, [9])
    print(real_data.shape)
    print(valid_data.shape)
    num_batches = num_batches // 10 * 9
    num_valid_batches = num_batches // 10 * 1

    if is_pretrain:
        num_epoches = FLAGS.pretrain_epoches
    else:
        num_epoches = FLAGS.total_epoches
    D_loss_mean = 0.0
    G_loss_mean = 0.0
    log_counter = 0
    # to evaluate time cost
    start_time = time.time()
    for epoch_id in range(num_epoches):
        # shuffle the data
        shuffled_indexes = np.random.permutation(real_data.shape[0])
        real_data = real_data[shuffled_indexes]

        batch_id = 0
        while batch_id < num_batches - FLAGS.num_train_D:
            real_data_batch = None
            if epoch_id < FLAGS.num_pretrain_D or (epoch_id + 1) % FLAGS.freq_train_D == 0:
                num_train_D = num_batches * 5  # TODO
            else:
                num_train_D = FLAGS.num_train_D
            for id_ in range(num_train_D):
                if (id_ + 1) % num_batches == 0:
                    # shuffle the data
                    shuffled_indexes = np.random.permutation(
                        real_data.shape[0])
                    real_data = real_data[shuffled_indexes]
                # make sure not exceed the boundary
                data_idx = batch_id * \
                    FLAGS.batch_size % (real_data.shape[0] - FLAGS.batch_size)
                # data
                real_data_batch = real_data[data_idx:data_idx +
                                            FLAGS.batch_size]
                # train D
                D_loss_mean, global_steps = model.D_step(
                    sess, z_samples(), real_data_batch, is_pretrain)
                batch_id += 1
                log_counter += 1
                # log validation loss
                data_idx = global_steps * \
                    FLAGS.batch_size % (valid_data.shape[0] - FLAGS.batch_size)
                valid_data_batch = valid_data[data_idx:data_idx +
                                              FLAGS.batch_size]
                D_valid_loss_mean = model.D_log_valid_loss(
                    sess, z_samples(), real_data_batch)
            # train G
            G_loss_mean, global_steps = model.G_step(
                sess, z_samples(), is_pretrain, real_data_batch)
            log_counter += 1

            # logging
            if log_counter >= FLAGS.log_freq:
                end_time = time.time()
                log_counter = 0
                print("%d, epoches, %d steps, mean D_loss: %f, mean D_valid_loss: %f, mean G_loss: %f, time cost: %f(sec)" %
                      (epoch_id,
                       global_steps,
                       D_loss_mean,
                       D_valid_loss_mean,
                       G_loss_mean,
                       (end_time - start_time)))
                start_time = time.time()  # save checkpoints
        # save model
        if (epoch_id % FLAGS.save_model_freq) == 0 or epoch_id == FLAGS.total_epoches - 1:
            save_path = saver.save(
                sess, FLAGS.checkpoints_dir + "model.ckpt",
                global_step=global_steps)
            print("Model saved in file: %s" % save_path)
        # plot generated sample
        if (epoch_id % FLAGS.save_result_freq) == 0 or epoch_id == FLAGS.total_epoches - 1:
            samples = model.generate(
                sess, z_samples(), is_pretrain, real_data_batch)
            # scale recovering
            samples = normer.recover_data(samples)
            # plot
            game_visualizer.plot_data(
                samples, FLAGS.seq_length, file_path=FLAGS.sample_dir + str(global_steps) + '.gif', if_save=True)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data and remove useless z dimension of players in data
        real_data = np.load(FLAGS.data_path)
        print('real_data.shape', real_data.shape)

        # normalization
        normer = Norm(real_data)
        real_data = normer.get_normed_data()[:, :FLAGS.seq_length, :]
        print(real_data.shape)

        # number of batches
        num_batches = real_data.shape[0] // FLAGS.batch_size
        # config setting
        config = TrainingConfig()
        config.show()
        # model
        model = model_rnn_wgan.RNN_WGAN(config, graph)
        init = tf.global_variables_initializer()
        # saver for later restore
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print('successfully restore model from checkpoint: %s' %
                      (FLAGS.restore_path))
            # pre-training
            if FLAGS.pretrain_epoches > 0:
                training(sess, model, real_data, num_batches,
                         saver, normer, is_pretrain=True)

            # training
            training(sess, model, real_data, num_batches, saver, normer)


if __name__ == '__main__':
    if FLAGS.restore_path is None:
        # when not restore, remove follows (old) for new training
        if os.path.exists(FLAGS.log_dir):
            shutil.rmtree(FLAGS.log_dir)
            print('rm -rf "%s" complete!' % FLAGS.log_dir)
        if os.path.exists(FLAGS.checkpoints_dir):
            shutil.rmtree(FLAGS.checkpoints_dir)
            print('rm -rf "%s" complete!' % FLAGS.checkpoints_dir)
        if os.path.exists(FLAGS.sample_dir):
            shutil.rmtree(FLAGS.sample_dir)
            print('rm -rf "%s" complete!' % FLAGS.sample_dir)
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    tf.app.run()
