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
tf.app.flags.DEFINE_integer('latent_dims', 100,
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
                           3 -> to visulize weight \
                           4 -> to analize code, only change first dimension for comparison \
                           5 -> to calculate hueristic score on selected result\
                           6 -> to draw different length result \
                           7 -> to draw feature map \
                           8 -> to find high-openshot-penalty data in real dataset")

# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def z_samples(batch_size):
    return np.random.normal(
        0., 1., size=[batch_size, FLAGS.latent_dims])


def mode_1(sess, graph, save_path, is_valid=FLAGS.is_valid):
    """ to collect results 
    Saved Result
    ------------
    results_A_fake_B : float, numpy ndarray, shape=[n_latents=100, n_conditions=128*9, length=100, features=23]
        Real A + Fake B
    results_A_real_B : float, numpy ndarray, shape=[n_conditions=128*9, length=100, features=23]
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
    """ to show diversity
    Saved Result
    ------------
    results_A_fake_B : float, numpy ndarray, shape=[latent_dims=10, n_latents=11, n_conditions=128, length=100, features=23]
        Real A + Fake B
    results_A_real_B : float, numpy ndarray, shape=[n_conditions=128, length=100, features=23]
        Real A + Real B
    results_critic_scores : float, numpy ndarray, shape=[latent_dims=10, n_latents=11, n_conditions=128]
        critic scores for each input data
    """
    n_latents = 100
    latent_dims = 1
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

    real_samples = target_data['B'][:512:4]
    real_conds = target_data['A'][:512:4]
    # generate result
    for target_dim in range(latent_dims):
        temp_critic_scores = []
        temp_A_fake_B = []
        for i in range(n_latents):
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
    # concat along with conditions dimension
    results_A_fake_B = np.concatenate(results_A_fake_B, axis=0)
    results_critic_scores = np.concatenate(results_critic_scores, axis=0)
    results_A = data_factory.recover_BALL_and_A(
        target_data['A'][:512:4])
    results_real_B = data_factory.recover_B(
        target_data['B'][:512:4])
    results_A_real_B = np.concatenate([results_A, results_real_B], axis=-1)
    # saved as numpy
    print(np.array(results_A_fake_B).shape)
    print(np.array(results_A_real_B).shape)
    print(np.array(results_critic_scores).shape)
    np.save(save_path + 'results_A_fake_B.npy',
            np.array(results_A_fake_B).astype(np.float32).reshape([latent_dims, n_latents, 128, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_A_real_B.npy',
            np.array(results_A_real_B).astype(np.float32).reshape([128, FLAGS.seq_length, 23]))
    np.save(save_path + 'results_critic_scores.npy',
            np.array(results_critic_scores).astype(np.float32).reshape([latent_dims, n_latents, 128]))
    print('!!Completely Saved!!')


def weight_vis(graph, save_path):
    """ to visulize weight
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
        conds_linear = __get_var_list('G_inference/conds_linear')
        latents_linear = __get_var_list('G_inference/latents_linear')
        print(conds_linear.shape)
        print(latents_linear.shape)
        conds_linear_result, latents_linear_result = sess.run(
            [conds_linear, latents_linear])
        trace = go.Heatmap(z=np.concatenate(
            [conds_linear_result, latents_linear_result], axis=0))
        data = [trace]
        plotly.offline.plot(data, filename=os.path.join(
            save_path, 'G_inference_input.html'))
    print('!!Completely Saved!!')


def mode_4(sess, graph, save_path, is_valid=FLAGS.is_valid):
    """ to analize code, only change first dimension for comparison
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


def mode_5(sess, graph, save_path):
    """ to calculate hueristic score on selected result
    """
    NORMAL_C_ID = [154, 108, 32, 498, 2, 513, 263, 29, 439, 249, 504, 529, 24, 964, 641, 739, 214, 139, 819, 1078, 772, 349, 676, 1016, 582, 678, 39, 279,
                   918, 477, 809, 505, 896, 600, 564, 50, 810, 1132, 683, 578, 1131, 887, 621, 1097, 665, 528, 310, 631, 1102, 6, 945, 1020, 853, 490, 64, 1002, 656]
    NORMAL_N_ID = [58, 5, 47, 66, 79, 21, 70, 54, 3, 59, 67, 59, 84, 38, 71, 62, 55, 86, 14, 83, 94, 97, 83, 27, 38, 68, 95,
                   26, 60, 2, 54, 46, 34, 75, 38, 4, 59, 87, 52, 44, 92, 28, 86, 71, 24, 28, 13, 70, 87, 44, 52, 25, 59, 61, 86, 16, 98]
    GOOD_C_ID = [976, 879, 293, 750, 908, 878, 831, 1038, 486, 268,
                 265, 252, 1143, 383, 956, 974, 199, 777, 585, 34, 932]
    GOOD_N_ID = [52, 16, 87, 43, 45, 66, 22, 77, 36,
                 50, 47, 9, 34, 9, 82, 42, 65, 43, 7, 29, 62]
    BEST_C_ID = [570, 517, 962, 1088, 35, 623, 1081, 33, 255, 571,
                 333, 990, 632, 431, 453, 196, 991, 267, 591, 902, 597, 646]
    BEST_N_ID = [22, 42, 76, 92, 12, 74, 92, 58, 69, 69,
                 23, 63, 89, 7, 74, 27, 12, 20, 35, 77, 62, 63]

    DUMMY_ID = np.zeros(shape=[28])
    ALL_C_ID = np.concatenate(
        [NORMAL_C_ID, GOOD_C_ID, BEST_C_ID, DUMMY_ID]).astype(np.int32)
    ALL_N_ID = np.concatenate(
        [NORMAL_N_ID, GOOD_N_ID, BEST_N_ID, DUMMY_ID]).astype(np.int32)
    print(ALL_C_ID.shape)
    print(ALL_N_ID.shape)
    fake_result_AB = np.load(
        'v3/2/collect/mode_1/results_A_fake_B.npy')[ALL_N_ID, ALL_C_ID]
    real_result_AB = np.load(
        'v3/2/collect/mode_1/results_A_real_B.npy')[ALL_C_ID]
    print(fake_result_AB.shape)
    print(real_result_AB.shape)

    # normalize
    real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
    print('real_data.shape', real_data.shape)
    data_factory = DataFactory(real_data)
    fake_result_AB = data_factory.normalize(fake_result_AB)
    real_result_AB = data_factory.normalize(real_result_AB)

    # placeholder tensor
    real_data_t = graph.get_tensor_by_name('real_data:0')
    matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
    # result tensor
    heuristic_penalty_pframe = graph.get_tensor_by_name(
        'Critic/C_inference/heuristic_penalty/Min:0')
    # 'Generator/G_loss/C_inference/linear_result/Reshape:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # real
    feed_dict = {
        real_data_t: real_result_AB[:, :, 13:23],
        matched_cond_t: real_result_AB[:, :, :13]
    }
    real_hp_pframe = sess.run(heuristic_penalty_pframe, feed_dict=feed_dict)

    # fake
    feed_dict = {
        real_data_t: fake_result_AB[:, :, 13:23],
        matched_cond_t: fake_result_AB[:, :, :13]
    }
    fake_hp_pframe = sess.run(heuristic_penalty_pframe, feed_dict=feed_dict)

    print(np.mean(real_hp_pframe[:100]))
    print(np.mean(fake_hp_pframe[:100]))
    print('!!Completely Saved!!')


def mode_6(sess, graph, save_path):
    """ to draw different length result
    """
    # normalize
    real_data = np.load(FLAGS.data_path)
    print('real_data.shape', real_data.shape)
    data_factory = DataFactory(real_data)
    target_data = np.load('FEATURES-7.npy')[:,:]
    team_AB = np.concatenate(
        [
            # ball
            target_data[:, :, 0, :3].reshape(
                [target_data.shape[0], target_data.shape[1], 1 * 3]),
            # team A players
            target_data[:, :, 1:6, :2].reshape(
                [target_data.shape[0], target_data.shape[1], 5 * 2]),
            # team B players
            target_data[:, :, 6:11, :2].reshape(
                [target_data.shape[0], target_data.shape[1], 5 * 2])
        ], axis=-1
    )
    team_AB = data_factory.normalize(team_AB)
    team_A = team_AB[:, :, :13]
    team_B = team_AB[:, :, 13:]
    # placeholder tensor
    latent_input_t = graph.get_tensor_by_name('latent_input:0')
    team_a_t = graph.get_tensor_by_name('team_a:0')
    # result tensor
    result_t = graph.get_tensor_by_name(
        'Generator/G_inference/conv_result/conv1d/Maximum:0')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # result collector
    latents = z_samples(team_AB.shape[0])
    feed_dict = {
        latent_input_t: latents,
        team_a_t: team_A
    }
    result_fake_B = sess.run(result_t, feed_dict=feed_dict)
    results_A_fake_B = np.concatenate([team_A, result_fake_B], axis=-1)
    results_A_fake_B = data_factory.recover_data(results_A_fake_B)
    for i in range(results_A_fake_B.shape[0]):
        game_visualizer.plot_data(
            results_A_fake_B[i], target_data.shape[1], file_path=save_path + str(i) + '.mp4', if_save=True)

    print('!!Completely Saved!!')


def mode_7(sess, graph, save_path):
    """ to draw feature map
    """
    # normalize
    real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
    print('real_data.shape', real_data.shape)
    data_factory = DataFactory(real_data)
    target_data = np.load('FEATURES-6.npy')[:6]
    team_AB = np.concatenate(
        [
            # ball
            target_data[:, :, 0, :3].reshape(
                [target_data.shape[0], target_data.shape[1], 1 * 3]),
            # team A players
            target_data[:, :, 1:6, :2].reshape(
                [target_data.shape[0], target_data.shape[1], 5 * 2]),
            # team B players
            target_data[:, :, 6:11, :2].reshape(
                [target_data.shape[0], target_data.shape[1], 5 * 2])
        ], axis=-1
    )
    dummy_AB = np.zeros(shape=[128 - 6, 100, 23])
    team_AB = np.concatenate([team_AB, dummy_AB], axis=0)
    team_AB = data_factory.normalize(team_AB)
    team_A = team_AB[:, :, :13]
    team_B = team_AB[:, :, 13:]
    # placeholder tensor
    latent_input_t = graph.get_tensor_by_name('latent_input:0')
    team_a_t = graph.get_tensor_by_name('team_a:0')
    # result tensor
    conds_linear_t = graph.get_tensor_by_name(
        'Generator/G_inference/conds_linear/BiasAdd:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # result collector
    latents = np.concatenate([z_samples(1)
                              for i in range(FLAGS.batch_size)], axis=0)
    feed_dict = {
        latent_input_t: latents,
        team_a_t: team_A
    }
    conds_linear = sess.run(conds_linear_t, feed_dict=feed_dict)
    for i in range(6):
        trace = go.Heatmap(z=conds_linear[i])
        data = [trace]
        plotly.offline.plot(data, filename=os.path.join(
            save_path, 'G_conds_linear' + str(i) + '.html'))

    print('!!Completely Saved!!')


def mode_8(sess, graph, save_path):
    """ to find high-openshot-penalty data in 1000 real data
    """
    real_data = np.load(FLAGS.data_path)[:, :FLAGS.seq_length, :, :]
    print('real_data.shape', real_data.shape)
    data_factory = DataFactory(real_data)
    train_data, valid_data = data_factory.fetch_data()
    # placeholder tensor
    real_data_t = graph.get_tensor_by_name('real_data:0')
    matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
    # result tensor
    heuristic_penalty_pframe = graph.get_tensor_by_name(
        'Critic/C_inference/heuristic_penalty/Min:0')
    # 'Generator/G_loss/C_inference/linear_result/Reshape:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    real_hp_pframe_all = []
    for batch_id in range(train_data['A'].shape[0] // FLAGS.batch_size):
        index_id = batch_id * FLAGS.batch_size
        real_data = train_data['B'][index_id:index_id + FLAGS.batch_size]
        cond_data = train_data['A'][index_id:index_id + FLAGS.batch_size]
        # real
        feed_dict = {
            real_data_t: real_data,
            matched_cond_t: cond_data
        }
        real_hp_pframe = sess.run(
            heuristic_penalty_pframe, feed_dict=feed_dict)
        real_hp_pframe_all.append(real_hp_pframe)
    real_hp_pframe_all = np.concatenate(real_hp_pframe_all, axis=0)
    print(real_hp_pframe_all.shape)
    real_hp_pdata = np.mean(real_hp_pframe_all, axis=1)
    mean_ = np.mean(real_hp_pdata)
    std_ = np.std(real_hp_pdata)
    print(mean_)
    print(std_)

    concat_AB = np.concatenate(
        [train_data['A'], train_data['B']], axis=-1)
    recoverd = data_factory.recover_data(concat_AB)
    for i, v in enumerate(real_hp_pdata):
        if v > (mean_ + 2 * std_):
            print('bad', i, v)
            game_visualizer.plot_data(
                recoverd[i], recoverd.shape[1], file_path=save_path + 'bad_' + str(i) + '_' + str(v) + '.mp4', if_save=True)
        if v < 0.0025:
            print('good', i, v)
            game_visualizer.plot_data(
                recoverd[i], recoverd.shape[1], file_path=save_path + 'good_' + str(i) + '_' + str(v) + '.mp4', if_save=True)

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
            elif FLAGS.mode == 4:
                mode_4(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_4/'))
            elif FLAGS.mode == 5:
                mode_5(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_5/'))
            elif FLAGS.mode == 6:
                mode_6(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_6/'))
            elif FLAGS.mode == 7:
                mode_7(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_7/'))
            elif FLAGS.mode == 8:
                mode_8(sess, graph, save_path=os.path.join(
                    COLLECT_PATH, 'mode_8/'))


if __name__ == '__main__':
    assert FLAGS.restore_path is not None
    assert FLAGS.mode is not None
    assert FLAGS.folder_path is not None
    global COLLECT_PATH
    COLLECT_PATH = os.path.join(FLAGS.folder_path, 'collect/')
    if not os.path.exists(COLLECT_PATH):
        os.makedirs(COLLECT_PATH)
    tf.app.run()
