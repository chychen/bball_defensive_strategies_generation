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
import plotly
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go

FLAGS = tf.app.flags.FLAGS


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
# model parameters
tf.app.flags.DEFINE_string('gpus', '0',
                           "define visible gpus")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "batch size")
tf.app.flags.DEFINE_integer('latent_dims', 10,
                            "latent_dims")
# collect mode
tf.app.flags.DEFINE_integer('n_latents', 100,
                            "n_latents")
tf.app.flags.DEFINE_integer('n_conditions', 128 * 9,  # because there are 9 batch_size data amount in validation set
                            "n_conditions")
tf.app.flags.DEFINE_bool('is_valid', True,
                         "is_valid")
tf.app.flags.DEFINE_integer('mode', None,
                            "mode to collect, \
                           1 -> to collect results \
                           2 -> to show diversity \
                           3 -> weight visualization")

# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def z_samples(batch_size):
    return np.random.normal(
        0., 1., size=[batch_size, FLAGS.latent_dims])


def mode_1(sess, graph, save_path, is_valid=FLAGS.is_valid):
    """ collect results
    Saved Result
    ------------
    results_A_fake_B : float, numpy ndarray, shape=[n_latents=100, n_conditions=128*9, length=100, features=23]
        Real A + Fake B
    results_A_real_B : float, numpy ndarray, shape=[n_latents=100, n_conditions=128*9, length=100, features=23]
        Real A + Real B
    results_critic_scores : float, numpy ndarray, shape=[n_latents=100, n_conditions=128*9]
        critic scores for each input data
    """
    # placeholder tensor
    latent_input_t = graph.get_tensor_by_name('latent_input:0')
    team_a_t = graph.get_tensor_by_name('team_a:0')
    G_samples_t = graph.get_tensor_by_name('G_samples:0')
    matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
    # result tensor
    result_t = graph.get_tensor_by_name(
        'Generator/G_inference/conv_result/conv1d/Maximum:0')
    critic_scores_t = graph.get_tensor_by_name(
        'Critic/C_inference_1/linear_result/BiasAdd:0')
    # 'Generator/G_loss/C_inference/linear_result/Reshape:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
    print('real_data.shape', real_data.shape)
    # normalize
    data_factory = DataFactory(real_data)
    # result collector
    results_A_fake_B = []
    results_A_real_B = []
    results_critic_scores = []

    # shuffle the data
    train_data, valid_data = data_factory.fetch_data()
    if is_valid:
        target_data = valid_data
    else:
        target_data = train_data

    for idx in range(0, FLAGS.n_conditions, FLAGS.batch_size):
        real_samples = target_data['B'][idx:idx + FLAGS.batch_size]
        real_conds = target_data['A'][idx:idx + FLAGS.batch_size]
        # generate result
        temp_critic_scores = []
        temp_A_fake_B = []
        for i in range(FLAGS.n_latents):
            latents = z_samples(FLAGS.batch_size)
            feed_dict = {
                latent_input_t: latents,
                team_a_t: real_conds
            }
            result = sess.run(
                result_t, feed_dict=feed_dict)
            feed_dict = {
                G_samples_t: result,
                matched_cond_t: real_conds
            }
            critic_scores = sess.run(
                critic_scores_t, feed_dict=feed_dict)
            temp_A_fake_B.append(data_factory.recover_data(
                np.concatenate([real_conds, result], axis=-1)))
            temp_critic_scores.append(critic_scores)
        results_A_fake_B.append(temp_A_fake_B)
        results_critic_scores.append(temp_critic_scores)
    # concat along with conditions dimension (axis=1)
    results_A_fake_B = np.concatenate(results_A_fake_B, axis=1)
    results_critic_scores = np.concatenate(results_critic_scores, axis=1)
    results_A = data_factory.recover_BALL_and_A(
        target_data['A'][:FLAGS.n_conditions])
    results_real_B = data_factory.recover_B(
        target_data['B'][:FLAGS.n_conditions])
    results_A_real_B = np.concatenate([results_A, results_real_B], axis=-1)
    # saved as numpy
    print(np.array(results_A_fake_B).shape)
    print(np.array(results_A_real_B).shape)
    print(np.array(results_critic_scores).shape)
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32).reshape([FLAGS.n_latents, FLAGS.n_conditions, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_A_real_B.npy',
            np.array(results_A_real_B).astype(np.float32).reshape([FLAGS.n_conditions, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([FLAGS.n_latents, FLAGS.n_conditions]))
    print('!!Completely Saved!!')


def mode_2(sess, graph, save_path, is_valid=FLAGS.is_valid):
    """ to show diversity, only changing first dimension
    Saved Result
    ------------
    results_A_fake_B : float, numpy ndarray, shape=[n_latents=11, n_conditions=128*9, length=100, features=23]
        Real A + Fake B
    results_A_real_B : float, numpy ndarray, shape=[n_latents=11, n_conditions=128*9, length=100, features=23]
        Real A + Real B
    results_critic_scores : float, numpy ndarray, shape=[n_latents=11, n_conditions=128*9]
        critic scores for each input data
    """
    target_dims = 0
    n_latents = 11

    # placeholder tensor
    latent_input_t = graph.get_tensor_by_name('latent_input:0')
    team_a_t = graph.get_tensor_by_name('team_a:0')
    G_samples_t = graph.get_tensor_by_name('G_samples:0')
    matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
    # result tensor
    result_t = graph.get_tensor_by_name(
        'Generator/G_inference/conv_result/conv1d/Maximum:0')
    critic_scores_t = graph.get_tensor_by_name(
        'Critic/C_inference_1/linear_result/BiasAdd:0')
    # 'Generator/G_loss/C_inference/linear_result/Reshape:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
    print('real_data.shape', real_data.shape)
    # normalize
    data_factory = DataFactory(real_data)
    # result collector
    results_A_fake_B = []
    results_A_real_B = []
    results_critic_scores = []

    # shuffle the data
    train_data, valid_data = data_factory.fetch_data()
    if is_valid:
        target_data = valid_data
    else:
        target_data = train_data
    latents = z_samples(FLAGS.batch_size)
    for idx in range(0, FLAGS.n_conditions, FLAGS.batch_size):
        real_samples = target_data['B'][idx:idx + FLAGS.batch_size]
        real_conds = target_data['A'][idx:idx + FLAGS.batch_size]
        # generate result
        temp_critic_scores = []
        temp_A_fake_B = []
        for i in range(n_latents):
            latents[:, target_dims] = -2.5 + 0.5 * i
            feed_dict = {
                latent_input_t: latents,
                team_a_t: real_conds
            }
            result = sess.run(
                result_t, feed_dict=feed_dict)
            feed_dict = {
                G_samples_t: result,
                matched_cond_t: real_conds
            }
            critic_scores = sess.run(
                critic_scores_t, feed_dict=feed_dict)
            temp_A_fake_B.append(data_factory.recover_data(
                np.concatenate([real_conds, result], axis=-1)))
            temp_critic_scores.append(critic_scores)
        results_A_fake_B.append(temp_A_fake_B)
        results_critic_scores.append(temp_critic_scores)
    # concat along with conditions dimension (axis=1)
    results_A_fake_B = np.concatenate(results_A_fake_B, axis=1)
    results_critic_scores = np.concatenate(results_critic_scores, axis=1)
    results_A = data_factory.recover_BALL_and_A(
        target_data['A'][:FLAGS.n_conditions])
    results_real_B = data_factory.recover_B(
        target_data['B'][:FLAGS.n_conditions])
    results_A_real_B = np.concatenate([results_A, results_real_B], axis=-1)
    # saved as numpy
    print(np.array(results_A_fake_B).shape)
    print(np.array(results_A_real_B).shape)
    print(np.array(results_critic_scores).shape)
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32).reshape([n_latents, FLAGS.n_conditions, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_A_real_B.npy',
            np.array(results_A_real_B).astype(np.float32).reshape([FLAGS.n_conditions, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([n_latents, FLAGS.n_conditions]))
    print('!!Completely Saved!!')


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # sesstion config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
        with tf.Session(config=config) as sess:
            # restored
            saver.restore(sess, FLAGS.restore_path)
            # collect
            if FLAGS.mode == 1:
                mode_1(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_1/'))
            elif FLAGS.mode == 2:
                mode_2(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_2/'))
            elif FLAGS.mode == 3:
                weight_vis(graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_3/'))


if __name__ == '__main__':
    assert FLAGS.restore_path is not None
    assert FLAGS.mode is not None
    assert FLAGS.folder_path is not None
    global COLLECT_PATH
    COLLECT_PATH = os.path.join(FLAGS.folder_path, 'collect/')
    if not os.path.exists(COLLECT_PATH):
        os.makedirs(COLLECT_PATH)
    tf.app.run()
