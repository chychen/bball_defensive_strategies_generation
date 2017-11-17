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
tf.app.flags.DEFINE_integer('mode', None,
                            "mode to collect, \
                           1 -> to show diversity, only changing first dimension \
                           2 -> to cmp with other trained model, using the same conds and latents to gather 1280 results \
                           3 -> to show best result given same conditions, gathering 100 result by 100 latents per condition \
                           4 -> draw heat map on particular layers's weight")

# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def z_samples(batch_size):
    return np.random.normal(
        0., 1., size=[batch_size, FLAGS.latent_dims])


def mode_1(sess, graph, save_path):
    """ to show diversity, only changing first dimension 
    Saved Result
    ------------
    results_A : float, numpy ndarray, shape=[batch_size, length=100, features_A=13]
        conditional constraints of team A position sequence that generate results
    results_real_B : float, numpy ndarray, shape=[batch_size, length=100, features_B=10]
        original real of team B position sequence, used to compare with fake B
    results_fake_B : float, numpy ndarray, shape=[10, batch_size, length=100, features_B=10]
        generated results
    results_A_fake_B : float, numpy ndarray, shape=[batch_size, length=100, features=23]
        generated results with both conds and fake B
    results_critic_scores : float, numpy ndarray, shape=[10, batch_size]
        critic scores for each input data
    results_latent : float, numpy ndarray, shape=[10, batch_size, latent_dims]
        latent variables that generate the results

    Notes
    -----
    features_A=13 : ball(x,y,z)=3*1 + players(x,y)=2*5
    features_B=10 : players(x,y)=2*5
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
    results_A = []
    results_fake_B = []
    results_A_fake_B = []
    results_critic_scores = []
    results_real_B = []
    results_latent = []

    # shuffle the data
    train_data, valid_data = data_factory.shuffle()
    # one condition cmp different z
    real_samples = train_data['B'][0:FLAGS.batch_size]
    real_conds = train_data['A'][0:FLAGS.batch_size]
    # generate result
    latents_base = z_samples(FLAGS.batch_size)
    for i in range(10):
        latents = latents_base
        latents[:, 0] = -2.5 + i * 0.5
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
        results_A_fake_B.append(data_factory.recover_data(
            np.concatenate([real_conds, result], axis=-1)))
        result = data_factory.recover_B(result)
        results_fake_B.append(result)
        results_critic_scores.append(critic_scores)
        results_latent.append(latents)
    real_conds = data_factory.recover_BALL_and_A(real_conds)
    results_A = real_conds
    real_samples = data_factory.recover_B(real_samples)
    results_real_B = real_samples
    # saved as numpy
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32))
    np.save(save_path + 'results_A.npy',
            np.array(results_A).astype(np.float32))
    np.save(save_path + 'results_real_B.npy',
            np.array(results_real_B).astype(np.float32))
    np.save(save_path + 'results_fake_B.npy',
            np.array(results_fake_B).astype(np.float32))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([-1, FLAGS.batch_size]))
    # np.array(results_critic_scores).astype(np.float32))
    np.save(save_path + 'results_latent.npy',
            np.array(results_latent).astype(np.float32))
    print('!!Completely Saved!!')


def mode_2(sess, graph, save_path):
    """ to cmp with other trained model, using the same conds and latents to gather 1280 results
    Saved Result
    ------------
    results_A : float, numpy ndarray, shape=[batch_size, length=100, features_A=13]
        conditional constraints of team A position sequence that generate results
    results_real_B : float, numpy ndarray, shape=[batch_size, length=100, features_B=10]
        original real of team B position sequence, used to compare with fake B
    results_fake_B : float, numpy ndarray, shape=[number_diff_z, batch_size, length=100, features_B=10]
        generated results
    results_critic_scores : float, numpy ndarray, shape=[number_diff_z, batch_size]
        critic scores for each input data
    results_latent : float, numpy ndarray, shape=[number_diff_z, batch_size, latent_dims]
        latent variables that generate the results

    Notes
    -----
    features_A=13 : ball(x,y,z)=3*1 + players(x,y)=2*5
    features_B=10 : players(x,y)=2*5
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
    # load conditions and latents
    if not os.path.exists('co_latents.npy') or not os.path.exists('co_conditions.npy'):
        real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
        print('real_data.shape', real_data.shape)
        # normalize
        data_factory = DataFactory(real_data)
        train_data, _ = data_factory.shuffle()
        real_samples = train_data['B'][:1280]
        real_conds = train_data['A'][:1280]
        latents = z_samples(1280)
        np.save('co_real_samples.npy', real_samples)
        np.save('co_latents.npy', latents)
        np.save('co_conditions.npy', real_conds)
    else:
        real_samples = np.load('co_real_samples.npy')
        latents = np.load('co_latents.npy')
        real_conds = np.load('co_conditions.npy')

    # result collector
    results_A = []
    results_fake_B = []
    results_A_fake_B = []
    results_critic_scores = []
    results_real_B = []
    results_latent = []

    n_batch = 1280 // FLAGS.batch_size
    for batch_id in range(n_batch):
        idx = batch_id * FLAGS.batch_size
        real_samples = train_data['B'][idx:idx + FLAGS.batch_size]
        real_conds = train_data['A'][idx:idx + FLAGS.batch_size]
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
        results_A_fake_B.append(data_factory.recover_data(
            np.concatenate([real_conds, result], axis=-1)))
        result = data_factory.recover_B(result)
        results_fake_B.append(result)
        results_critic_scores.append(critic_scores)
        results_latent.append(latents)
        real_conds = data_factory.recover_BALL_and_A(real_conds)
        results_A.append(real_conds)
        real_samples = data_factory.recover_B(real_samples)
        results_real_B.append(real_samples)
    # saved as numpy
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32))
    np.save(save_path + 'results_A.npy',
            np.array(results_A).astype(np.float32))
    np.save(save_path + 'results_real_B.npy',
            np.array(results_real_B).astype(np.float32))
    np.save(save_path + 'results_fake_B.npy',
            np.array(results_fake_B).astype(np.float32))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([-1, FLAGS.batch_size]))
    # np.array(results_critic_scores).astype(np.float32))
    np.save(save_path + 'results_latent.npy',
            np.array(results_latent).astype(np.float32))
    print('!!Completely Saved!!')


def mode_3(sess, graph, save_path):
    """ to show best result given same conditions, gathering 100 result by 100 latents per condition"
    Saved Result
    ------------
    results_A : float, numpy ndarray, shape=[batch_size, length=100, features_A=13]
        conditional constraints of team A position sequence that generate results
    results_real_B : float, numpy ndarray, shape=[batch_size, length=100, features_B=10]
        original real of team B position sequence, used to compare with fake B
    results_fake_B : float, numpy ndarray, shape=[100, batch_size, length=100, features_B=10]
        generated results
    results_A_fake_B : float, numpy ndarray, shape=[batch_size, length=100, features=23]
        generated results with both conds and fake B
    results_critic_scores : float, numpy ndarray, shape=[100, batch_size]
        critic scores for each input data
    results_latent : float, numpy ndarray, shape=[100, batch_size, latent_dims]
        latent variables that generate the results

    Notes
    -----
    features_A=13 : ball(x,y,z)=3*1 + players(x,y)=2*5
    features_B=10 : players(x,y)=2*5
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
    results_A = []
    results_fake_B = []
    results_A_fake_B = []
    results_critic_scores = []
    results_real_B = []
    results_latent = []

    # shuffle the data
    train_data, valid_data = data_factory.shuffle()
    # one condition cmp different z
    real_samples = train_data['B'][0:FLAGS.batch_size]
    real_conds = train_data['A'][0:FLAGS.batch_size]
    # generate result
    for i in range(100):
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
        results_A_fake_B.append(data_factory.recover_data(
            np.concatenate([real_conds, result], axis=-1)))
        result = data_factory.recover_B(result)
        results_fake_B.append(result)
        results_critic_scores.append(critic_scores)
        results_latent.append(latents)
    real_conds = data_factory.recover_BALL_and_A(real_conds)
    results_A = real_conds
    real_samples = data_factory.recover_B(real_samples)
    results_real_B = real_samples
    # saved as numpy
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32))
    np.save(save_path + 'results_A.npy',
            np.array(results_A).astype(np.float32))
    np.save(save_path + 'results_real_B.npy',
            np.array(results_real_B).astype(np.float32))
    np.save(save_path + 'results_fake_B.npy',
            np.array(results_fake_B).astype(np.float32))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([-1, FLAGS.batch_size]))
    # np.array(results_critic_scores).astype(np.float32))
    np.save(save_path + 'results_latent.npy',
            np.array(results_latent).astype(np.float32))
    print('!!Completely Saved!!')


def weight_vis(graph, save_path):
    """ draw heat map on particular layers's weight
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def __get_var_list(tag):
        """ to get both Generator's and Discriminator's trainable variables
        and add trainable variables into histogram
        """
        trainable_V = tf.trainable_variables()
        for _, v in enumerate(trainable_V):
            if tag in v.name:
                return v
    # sesstion config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
    with tf.Session(config=config) as sess:
        # restored
        saver.restore(sess, FLAGS.restore_path)
        # target tensor
        theta = __get_var_list('C_inference/conv_input')
        print(theta.shape)
        trace = go.Heatmap(z=sess.run(theta[0]))
        data = [trace]
        plotly.offline.plot(data, filename='C_inference_conv_input.html')


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
                mode_3(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_3/'))
            elif FLAGS.mode == 4:
                weight_vis(graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_4/'))


if __name__ == '__main__':
    assert FLAGS.restore_path is not None
    assert FLAGS.mode is not None
    assert FLAGS.folder_path is not None
    global COLLECT_PATH
    COLLECT_PATH = os.path.join(FLAGS.folder_path, 'collect/')
    if not os.path.exists(COLLECT_PATH):
        os.makedirs(COLLECT_PATH)
    tf.app.run()
