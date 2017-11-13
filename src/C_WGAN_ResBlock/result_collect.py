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
# TODO read params from checkpoints directory (hyper_parameters.json)
tf.app.flags.DEFINE_string('folder_path', 'v1/3',
                           "summary directory")
tf.app.flags.DEFINE_string('data_path', '../../data/FEATURES-4.npy',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# input parameters
tf.app.flags.DEFINE_integer('seq_length', 100,
                            "the maximum length of one training data")
tf.app.flags.DEFINE_integer('latent_dims', 10,
                            "dimensions of latant variable")
# model parameters
tf.app.flags.DEFINE_string('gpus', '0',
                           "define visible gpus")
tf.app.flags.DEFINE_integer('batch_size', 256,
                            "batch size")
tf.app.flags.DEFINE_integer('number_diff_z', 100,
                            "number of different conditions of team A")

COLLECT_PATH = os.path.join(FLAGS.folder_path, 'collect/')
# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def z_samples():
    return np.random.normal(
        0., 1., size=[FLAGS.batch_size, FLAGS.latent_dims])


def collecting(data_factory, graph):
    """ collect result
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
    # result collector
    results_A = []
    results_fake_B = []
    results_critic_scores = []
    results_real_B = []
    results_latent = []
    # sesstion config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
    with tf.Session(config=config) as sess:
        # restored
        saver.restore(sess, FLAGS.restore_path)
        # placeholder tensor
        latent_input_t = graph.get_tensor_by_name('latent_input:0')
        team_a_t = graph.get_tensor_by_name('team_a:0')
        real_data_t = graph.get_tensor_by_name('real_data:0')
        matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
        # result tensor
        result_t = graph.get_tensor_by_name(
            'Generator/G_inference/conv_result/conv1d/Maximum:0')
        critic_scores_t = graph.get_tensor_by_name(
            'Critic/C_inference/linear_result/BiasAdd:0')
        # 'Generator/G_loss/C_inference/linear_result/Reshape:0')

        # shuffle the data
        train_data, valid_data = data_factory.shuffle()
        real_samples = train_data['B'][0:FLAGS.batch_size]
        real_conds = train_data['A'][0:FLAGS.batch_size]
        # generate result
        # latents_base = z_samples()
        for i in range(FLAGS.number_diff_z):
            # latents = latents_base
            # latents[:, 0] = -5 + i
            latents = z_samples()
            feed_dict = {
                latent_input_t: latents,
                team_a_t: real_conds,
                real_data_t: real_samples,
                matched_cond_t: real_conds
            }
            result, critic_scores = sess.run(
                [result_t, critic_scores_t], feed_dict=feed_dict)
            result = data_factory.recover_B(result)
            results_fake_B.append(result)
            results_critic_scores.append(critic_scores)
            results_latent.append(latents)
        real_conds = data_factory.recover_BALL_and_A(real_conds)
        results_A = real_conds
        real_samples = data_factory.recover_B(real_samples)
        results_real_B = real_samples
    # saved as numpy
    np.save(COLLECT_PATH + 'results_A.npy', results_A.astype(np.float32))
    np.save(COLLECT_PATH + 'results_real_B.npy',
            results_real_B.astype(np.float32))
    np.save(COLLECT_PATH + 'results_fake_B.npy',
            np.array(results_fake_B).astype(np.float32))
    np.save(COLLECT_PATH + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([FLAGS.number_diff_z, FLAGS.batch_size]))
    # np.array(results_critic_scores).astype(np.float32))
    np.save(COLLECT_PATH + 'results_latent.npy',
            np.array(results_latent).astype(np.float32))
    print('!!Completely Saved!!')


# def weight_vis(graph):
#     """ weight_vis
#     """
#     def __get_var_list(tag):
#         """ to get both Generator's and Discriminator's trainable variables
#         and add trainable variables into histogram
#         """
#         trainable_V = tf.trainable_variables()
#         theta = []
#         for _, v in enumerate(trainable_V):
#             if tag in v.name:
#                 theta.append(v)
#         return theta
#     # sesstion config
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     saver = tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
#     with tf.Session(config=config) as sess:
#         # restored
#         saver.restore(sess, FLAGS.restore_path)
#         # target tensor
#         theta = __get_var_list('G/linear/weight')
#         trace = go.Heatmap(z=sess.run(theta[0]))
#         data = [trace]
#         plotly.offline.plot(data, filename='G_linear_weight.html')


def main(_):
    with tf.get_default_graph().as_default() as graph:
        real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
        print('real_data.shape', real_data.shape)
        # normalize
        data_factory = DataFactory(real_data)
        train_data, valid_data = data_factory.fetch_data()
        print(train_data['A'].shape)
        print(valid_data['A'].shape)
        # collect
        collecting(data_factory, graph)

        # weight_vis(graph)


if __name__ == '__main__':
    assert FLAGS.restore_path is not None
    if not os.path.exists(COLLECT_PATH):
        os.makedirs(COLLECT_PATH)
    tf.app.run()
