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


FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string('log_dir', 'v1/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v1/checkpoints/',
                           "checkpoints dir")
tf.app.flags.DEFINE_string('sample_dir', 'v1/sample/',
                           "directory to save generative result")
tf.app.flags.DEFINE_string('data_path', '../data/NBA-TEAM1.npy',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# input parameters
tf.app.flags.DEFINE_integer('seq_length', 300,
                            "the maximum length of one training data")
tf.app.flags.DEFINE_integer('num_features', 23,
                            "3 (ball x y z) + 10 (players) * 2 (x and y)")
tf.app.flags.DEFINE_integer('latent_dims', 11,
                            "dimensions of latant variable")
# training parameters
tf.app.flags.DEFINE_integer('total_epoches', 500,
                            "num of ephoches")
tf.app.flags.DEFINE_integer('num_train_D', 5,
                            "num of times of training D before train G")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            "batch size")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          "learning rate")
tf.app.flags.DEFINE_integer('hidden_size', 110,
                            "hidden size of LSTM")
tf.app.flags.DEFINE_integer('rnn_layers', 1,
                            "num of layers for rnn")
tf.app.flags.DEFINE_integer('save_freq', 25,
                            "num of epoches to save model")
tf.app.flags.DEFINE_float('penalty_lambda', 10.0,
                          "regularization parameter of wGAN loss function")


class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self):
        self.total_epoches = FLAGS.total_epoches
        self.num_train_D = FLAGS.num_train_D
        self.batch_size = FLAGS.batch_size
        self.log_dir = FLAGS.log_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.sample_dir = FLAGS.sample_dir
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.hidden_size = FLAGS.hidden_size
        self.rnn_layers = FLAGS.rnn_layers
        self.save_freq = FLAGS.save_freq
        self.seq_length = FLAGS.seq_length
        self.num_features = FLAGS.num_features
        self.latent_dims = FLAGS.latent_dims
        self.penalty_lambda = FLAGS.penalty_lambda

    def show(self):
        print("total_epoches:", self.total_epoches)
        print("num_train_D:", self.num_train_D)
        print("batch_size:", self.batch_size)
        print("log_dir:", self.log_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("sample_dir:", self.sample_dir)
        print("data_path:", self.data_path)
        print("learning_rate:", self.learning_rate)
        print("hidden_size:", self.hidden_size)
        print("rnn_layers:", self.rnn_layers)
        print("save_freq:", self.save_freq)
        print("seq_length:", self.seq_length)
        print("num_features:", self.num_features)
        print("latent_dims:", self.latent_dims)
        print("penalty_lambda:", self.penalty_lambda)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data and remove useless z dimension of players in data
        real_data = np.load(FLAGS.data_path)[:, :, [
            0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31]]
        print('real_data.shape', real_data.shape)
        # TODO data normalization
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
            # training
            for _ in range(FLAGS.total_epoches):
                # shuffle the data
                shuffled_indexes = np.random.permutation(real_data.shape[0])
                real_data = real_data[shuffled_indexes]

                batch_id = 0
                # TODO sample z from normal-distribution than uniform-distribution
                while batch_id <= num_batches - FLAGS.num_train_D:
                    # time cost evaluation
                    start_time = time.time()
                    # Discriminator
                    D_loss_sum = 0.0
                    for _ in range(FLAGS.num_train_D):
                        batch_idx = batch_id * FLAGS.batch_size
                        # data
                        real_data_batch = real_data[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                        z = np.random.uniform(
                            -1., 1., size=[FLAGS.batch_size, FLAGS.seq_length, FLAGS.latent_dims])
                        D_loss, global_steps = model.D_step(
                            sess, z, real_data_batch)
                        D_loss_sum += D_loss
                        batch_id += 1
                    # Generator
                    z = np.random.uniform(
                        -1., 1., size=[FLAGS.batch_size, FLAGS.seq_length, FLAGS.latent_dims])
                    G_loss, global_steps = model.G_step(sess, z)
                    # logging
                    end_time = time.time()
                    epoch_id = int(global_steps // num_batches)
                    print("%d epoches, %d steps, mean D_loss: %f, mean G_loss: %f, time cost: %f(sec/batch)" %
                          (epoch_id,
                           global_steps,
                           D_loss_sum / FLAGS.num_train_D,
                           G_loss,
                           (end_time - start_time) / FLAGS.num_train_D))
                # generate sample per epoch (one event, 300 frames)
                z = np.random.uniform(
                    -1., 1., size=[FLAGS.batch_size, FLAGS.seq_length, FLAGS.latent_dims])
                samples = model.generate(sess, z)
                game_visualizer.plot_data(
                    samples, FLAGS.seq_length, file_path=FLAGS.sample_dir + str(epoch_id) + '.gif', if_save=True)
                # save checkpoints
                if (epoch_id % FLAGS.save_freq) == 0:
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir + "model.ckpt",
                        global_step=global_steps)
                    print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    if FLAGS.restore_path is None:
        # when not restore, remove follows for new training
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
