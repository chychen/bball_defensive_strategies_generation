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
import json
import numpy as np
import tensorflow as tf
import game_visualizer
from utils import DataFactory
from Generator import G_MODEL
from Critic import C_MODEL
FLAGS = tf.app.flags.FLAGS

# comment
tf.app.flags.DEFINE_string('comment', None,
                           "(required) what would you like to test?")
# path parameters
tf.app.flags.DEFINE_string('folder_path', None,
                           "summary directory")
tf.app.flags.DEFINE_string('data_path', '../../data/FEATURES-4.npy',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# input parameters
tf.app.flags.DEFINE_integer('seq_length', 100,
                            "the maximum length of one training data")
tf.app.flags.DEFINE_integer('latent_dims', 100,
                            "dimensions of latant variable")
# training parameters
tf.app.flags.DEFINE_string('gpus', '0',
                           "define visible gpus")
tf.app.flags.DEFINE_integer('total_epoches', 5000,
                            "num of ephoches")
tf.app.flags.DEFINE_integer('num_train_D', 5,
                            "num of times of training D before train G")
tf.app.flags.DEFINE_integer('num_pretrain_D', 10,
                            "num of ephoch to train D before train G")
tf.app.flags.DEFINE_integer('freq_train_D', 10,
                            "freqence of num ephoch to train D more")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "batch size")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          "learning rate")
tf.app.flags.DEFINE_float('penalty_lambda', 10.0,
                          "regularization parameter of wGAN loss function")
tf.app.flags.DEFINE_float('latent_penalty_lambda', 1.0,
                          "regularization for latent's weight")
tf.app.flags.DEFINE_integer('n_resblock', 4,
                            "number of resblock for Generator and Critic")
tf.app.flags.DEFINE_bool('if_handcraft_features', False,
                         "if_handcraft_features")
tf.app.flags.DEFINE_bool('if_feed_extra_info', True,
                         "if_feed_extra_info, e.g. basket position")
tf.app.flags.DEFINE_float('residual_alpha', 1.0,
                          "residual block = F(x) * residual_alpha + x")
tf.app.flags.DEFINE_float('leaky_relu_alpha', 0.2,
                          "tf.maximum(x, leaky_relu_alpha * x)")
tf.app.flags.DEFINE_float('heuristic_penalty_lambda', 50.0,
                          "heuristic_penalty_lambda")
tf.app.flags.DEFINE_bool('if_use_mismatched', False,
                         "if True, negative scores = mean of (fake_scores + mismatched_scores)")
tf.app.flags.DEFINE_bool('if_trainable_lambda', False,
                         "if_trainable_lambda, if True: init=10.0")
tf.app.flags.DEFINE_integer('n_filters', 256,
                            "number of filters in all ConV")
# logging
tf.app.flags.DEFINE_integer('save_model_freq', 100,
                            "num of epoches to save model")
tf.app.flags.DEFINE_integer('save_result_freq', 100,
                            "num of epoches to save gif")
tf.app.flags.DEFINE_integer('log_freq', 1000,
                            "num of steps to log")

# PATH
LOG_PATH = os.path.join(FLAGS.folder_path, 'log/')
CHECKPOINTS_PATH = os.path.join(FLAGS.folder_path, 'checkpoints/')
SAMPLE_PATH = os.path.join(FLAGS.folder_path, 'sample/')
# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self):
        self.folder_path = FLAGS.folder_path
        self.total_epoches = FLAGS.total_epoches
        self.batch_size = FLAGS.batch_size
        self.log_dir = LOG_PATH
        self.checkpoints_dir = CHECKPOINTS_PATH
        self.sample_dir = SAMPLE_PATH
        self.data_path = FLAGS.data_path
        self.learning_rate = FLAGS.learning_rate
        self.save_model_freq = FLAGS.save_model_freq
        self.save_result_freq = FLAGS.save_result_freq
        self.log_freq = FLAGS.log_freq
        self.seq_length = FLAGS.seq_length
        self.latent_dims = FLAGS.latent_dims
        self.penalty_lambda = FLAGS.penalty_lambda
        self.latent_penalty_lambda = FLAGS.latent_penalty_lambda
        self.num_train_D = FLAGS.num_train_D
        self.num_pretrain_D = FLAGS.num_pretrain_D
        self.freq_train_D = FLAGS.freq_train_D
        self.n_resblock = FLAGS.n_resblock
        self.if_handcraft_features = FLAGS.if_handcraft_features
        self.if_feed_extra_info = FLAGS.if_feed_extra_info
        self.residual_alpha = FLAGS.residual_alpha
        self.leaky_relu_alpha = FLAGS.leaky_relu_alpha
        self.heuristic_penalty_lambda = FLAGS.heuristic_penalty_lambda
        self.if_use_mismatched = FLAGS.if_use_mismatched
        self.if_trainable_lambda = FLAGS.if_trainable_lambda
        self.n_filters = FLAGS.n_filters
        with open(os.path.join(FLAGS.folder_path, 'hyper_parameters.json'), 'w') as outfile:
            json.dump(FLAGS.__dict__['__flags'], outfile)

    def show(self):
        print(FLAGS.__dict__['__flags'])


def z_samples():
    return np.random.normal(
        0., 1., size=[FLAGS.batch_size, FLAGS.latent_dims])


def training(train_data, valid_data, data_factory, config, graph):
    """ training
    """
    # number of batches
    num_batches = train_data['A'].shape[0] // FLAGS.batch_size
    num_valid_batches = valid_data['A'].shape[0] // FLAGS.batch_size
    print('num_batches', num_batches)
    print('num_valid_batches', num_valid_batches)
    # model
    C = C_MODEL(config, graph)
    G = G_MODEL(config, C.inference, graph)
    init = tf.global_variables_initializer()
    # saver for later restore
    saver = tf.train.Saver(max_to_keep=0)  # 0 -> keep them all
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
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
            train_data, valid_data = data_factory.shuffle()

            batch_id = 0
            while batch_id < num_batches - FLAGS.num_train_D:
                real_data_batch = None
                if epoch_id < FLAGS.num_pretrain_D or (epoch_id + 1) % FLAGS.freq_train_D == 0:
                    num_train_D = num_batches
                else:
                    num_train_D = FLAGS.num_train_D
                for id_ in range(num_train_D):
                    # make sure not exceed the boundary
                    data_idx = batch_id * \
                        FLAGS.batch_size % (
                            train_data['B'].shape[0] - FLAGS.batch_size)
                    # data
                    real_samples = train_data['B'][data_idx:data_idx +
                                                   FLAGS.batch_size]
                    real_conds = train_data['A'][data_idx:data_idx +
                                                 FLAGS.batch_size]
                    # samples
                    fake_samples = G.generate(
                        sess, z_samples(), real_conds)
                    # train Critic
                    D_loss_mean, global_steps = C.step(
                        sess, fake_samples, real_samples, real_conds)
                    batch_id += 1
                    log_counter += 1

                    # log validation loss
                    data_idx = global_steps * \
                        FLAGS.batch_size % (
                            valid_data['B'].shape[0] - FLAGS.batch_size)
                    valid_real_samples = valid_data['B'][data_idx:data_idx +
                                                         FLAGS.batch_size]
                    valid_real_conds = valid_data['A'][data_idx:data_idx +
                                                       FLAGS.batch_size]
                    fake_samples = G.generate(
                        sess, z_samples(), valid_real_conds)
                    D_valid_loss_mean = C.log_valid_loss(
                        sess, fake_samples, valid_real_samples, valid_real_conds)

                # train G
                G_loss_mean, global_steps = G.step(
                    sess, z_samples(), real_conds)
                log_counter += 1

                # logging
                if log_counter >= FLAGS.log_freq:
                    end_time = time.time()
                    log_counter = 0
                    print("%d, epoches, %d steps, mean C_loss: %f, mean C_valid_loss: %f, mean G_loss: %f, time cost: %f(sec)" %
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
                    sess, CHECKPOINTS_PATH + "model.ckpt",
                    global_step=global_steps)
                print("Model saved in file: %s" % save_path)
            # plot generated sample
            if (epoch_id % FLAGS.save_result_freq) == 0 or epoch_id == FLAGS.total_epoches - 1:
                # fake
                samples = G.generate(sess, z_samples(), real_conds)
                concat_ = np.concatenate([real_conds, samples], axis=-1)
                fake_result = data_factory.recover_data(concat_)
                game_visualizer.plot_data(
                    fake_result[0], FLAGS.seq_length, file_path=SAMPLE_PATH + str(global_steps) + '_fake.mp4', if_save=True)
                # real
                concat_ = np.concatenate([real_conds, real_samples], axis=-1)
                real_result = data_factory.recover_data(concat_)
                game_visualizer.plot_data(
                    real_result[0], FLAGS.seq_length, file_path=SAMPLE_PATH + str(global_steps) + '_real.mp4', if_save=True)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
        print('real_data.shape', real_data.shape)
        # normalize
        data_factory = DataFactory(real_data)
        train_data, valid_data = data_factory.fetch_data()
        print(train_data['A'].shape)
        print(valid_data['A'].shape)
        # config setting
        config = TrainingConfig()
        config.show()
        # train
        training(train_data, valid_data, data_factory, config, graph)


if __name__ == '__main__':
    assert FLAGS.comment is not None, 'comment is required, please add it by --comment'
    assert FLAGS.folder_path is not None, 'folder_path is required, please add it by --folder_path'
    if FLAGS.restore_path is None:
        if os.path.exists(FLAGS.folder_path):
            ans = input('"%s" will be removed!! are you sure (y/N)? ' % FLAGS.folder_path)
            if ans == 'Y' or ans =='y':
                # when not restore, remove follows (old) for new training
                shutil.rmtree(FLAGS.folder_path)
                print('rm -rf "%s" complete!' % FLAGS.folder_path)
            else:
                exit()

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)
    if not os.path.exists(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)
    tf.app.run()
