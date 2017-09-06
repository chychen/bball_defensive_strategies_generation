from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import model_dcae


FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string('log_dir', 'v6/log/',
                           "summary directory")
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
tf.app.flags.DEFINE_integer('total_epoches', 1000,
                            "num of ephoches")
tf.app.flags.DEFINE_integer('num_train_D', 5,
                            "num of times of training D before train G")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "batch size")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          "learning rate")
tf.app.flags.DEFINE_integer('hidden_size', 110,
                            "hidden size of LSTM")
tf.app.flags.DEFINE_integer('rnn_layers', 1,
                            "num of layers for rnn")
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
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.hidden_size = FLAGS.hidden_size
        self.rnn_layers = FLAGS.rnn_layers
        self.seq_length = FLAGS.seq_length
        self.num_features = FLAGS.num_features
        self.latent_dims = FLAGS.latent_dims
        self.penalty_lambda = FLAGS.penalty_lambda

    def show(self):
        print("total_epoches:", self.total_epoches)
        print("num_train_D:", self.num_train_D)
        print("batch_size:", self.batch_size)
        print("log_dir:", self.log_dir)
        print("data_path:", self.data_path)
        print("learning_rate:", self.learning_rate)
        print("hidden_size:", self.hidden_size)
        print("rnn_layers:", self.rnn_layers)
        print("seq_length:", self.seq_length)
        print("num_features:", self.num_features)
        print("latent_dims:", self.latent_dims)
        print("penalty_lambda:", self.penalty_lambda)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        pass


if __name__ == '__main__':
    if os.path.exists('test_only'):
        shutil.rmtree('test_only')
        print('rm -rf "test_only" complete!')
    tf.app.run()
