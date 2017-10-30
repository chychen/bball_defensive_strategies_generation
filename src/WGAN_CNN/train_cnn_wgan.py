"""
data url: http://140.113.210.14:6006/NBA/data/F2.npy
data description: 
    event by envet, with 300 sequence for each. (about 75 seconds)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import game_visualizer
from utils_cnn import Norm
from Generator import G_MODEL
from Critic import C_MODEL

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string('log_dir', 'v21/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v21/checkpoints/',
                           "checkpoints dir")
tf.app.flags.DEFINE_string('sample_dir', 'v21/sample/',
                           "directory to save generative result")
tf.app.flags.DEFINE_string('data_path', '../../data/F2.npy',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# input parameters
tf.app.flags.DEFINE_integer('seq_length', 1,
                            "the maximum length of one training data")
tf.app.flags.DEFINE_integer('num_features', 23,
                            "3 (ball x y z) + 10 (players) * 2 (x and y) + 70 (player positions as 10 7-dims-one-hot)")
tf.app.flags.DEFINE_integer('latent_dims', 23,
                            "dimensions of latant variable")
# training parameters
tf.app.flags.DEFINE_integer('total_epoches', 1500,
                            "num of ephoches")
tf.app.flags.DEFINE_integer('num_train_D', 5,
                            "num of times of training D before train G")
tf.app.flags.DEFINE_integer('num_pretrain_D', 5,
                            "num of ephoch to train D before train G")
tf.app.flags.DEFINE_integer('freq_train_D', 51,
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
# logging
tf.app.flags.DEFINE_integer('save_model_freq', 50,
                            "num of epoches to save model")
tf.app.flags.DEFINE_integer('save_result_freq', 10,
                            "num of epoches to save gif")
tf.app.flags.DEFINE_integer('log_freq', 50,
                            "num of steps to log")
tf.app.flags.DEFINE_bool('if_log_histogram', False,
                         "whether to log histogram or not")


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
        self.num_train_D = FLAGS.num_train_D
        self.num_pretrain_D = FLAGS.num_pretrain_D
        self.freq_train_D = FLAGS.freq_train_D
        self.if_log_histogram = FLAGS.if_log_histogram

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
        print("num_train_D:", self.num_train_D)
        print("num_pretrain_D:", self.num_pretrain_D)
        print("freq_train_D:", self.freq_train_D)
        print("if_log_histogram:", self.if_log_histogram)


def z_samples():
    # # TODO sample z from normal-distribution
    return np.random.uniform(
        0.0, 1.0, size=[FLAGS.batch_size, FLAGS.seq_length, FLAGS.latent_dims])


def training(real_data, normer, config, graph):
    """ training
    """
    # number of batches
    num_batches = real_data.shape[0] // FLAGS.batch_size
    shuffled_indexes = np.random.permutation(real_data.shape[0])
    real_data = real_data[shuffled_indexes]
    real_data, valid_data = np.split(real_data, [real_data.shape[0] // 9])
    print(real_data.shape)
    print(valid_data.shape)
    num_batches = num_batches // 10 * 9
    num_valid_batches = num_batches // 10 * 1
    # model
    C = C_MODEL(config, graph)
    G = G_MODEL(config, C.inference, graph)
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

        D_loss_mean = 0.0
        D_valid_loss_mean = 0.0
        G_loss_mean = 0.0
        log_counter = 0
        # to evaluate time cost
        start_time = time.time()
        for epoch_id in range(FLAGS.total_epoches):
            # shuffle the data
            shuffled_indexes = np.random.permutation(real_data.shape[0])
            real_data = real_data[shuffled_indexes]
            shuffled_indexes = np.random.permutation(valid_data.shape[0])
            valid_data = valid_data[shuffled_indexes]

            batch_id = 0
            while batch_id < num_batches - FLAGS.num_train_D:
                real_data_batch = None
                if epoch_id < FLAGS.num_pretrain_D or (epoch_id + 1) % FLAGS.freq_train_D == 0:
                    num_train_D = num_batches * 5  # TODO
                else:
                    num_train_D = FLAGS.num_train_D
                for id_ in range(num_train_D):
                    # make sure not exceed the boundary
                    data_idx = batch_id * \
                        FLAGS.batch_size % (
                            real_data.shape[0] - FLAGS.batch_size)
                    # data
                    real_samples = real_data[data_idx:data_idx +
                                             FLAGS.batch_size]
                    # samples
                    fake_samples = G.generate(sess, z_samples())
                    real_samples = normer.format_discrete_map(real_samples)
                    # train Critic
                    D_loss_mean, global_steps = C.step(
                        sess, fake_samples, real_samples)
                    batch_id += 1
                    log_counter += 1

                    # # log validation loss
                    # data_idx = global_steps * \
                    #     FLAGS.batch_size % (
                    #         valid_data.shape[0] - FLAGS.batch_size)
                    # valid_real_samples = valid_data[data_idx:data_idx +
                    #                                 FLAGS.batch_size]
                    # valid_real_samples = normer.format_discrete_map(valid_real_samples)
                    # D_valid_loss_mean = C.D_log_valid_loss(
                    #     sess, fake_samples, valid_real_samples)

                # train G
                G_loss_mean, global_steps = G.step(sess, z_samples())
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
                samples = G.generate(sess, z_samples())
                # scale recovering TODO
                samples = normer.map_2_position(samples)
                # plot
                game_visualizer.plot_data(
                    samples[0:], FLAGS.seq_length, file_path=FLAGS.sample_dir + str(global_steps) + '.gif', if_save=True)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data and remove useless z dimension of players in data
        real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :]
        print('real_data.shape', real_data.shape)

        # TODO normalization
        normer = Norm(real_data)
        # real_data = normer.get_normed_data()[:, :FLAGS.seq_length, :]
        print(real_data.shape)

        # config setting
        config = TrainingConfig()
        config.show()
        # train
        training(real_data, normer, config, graph)


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
