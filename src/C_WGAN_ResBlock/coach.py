"""
restore one critic as coach to serach best hyper-parameters
<Deprecated>
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
# path parameters
tf.app.flags.DEFINE_string('folder_path', None,
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# model parameters
tf.app.flags.DEFINE_string('gpus', '0',
                           "define visible gpus")

SCORE_TABLE_PATH = os.path.join(FLAGS.folder_path, 'score_table.json')
# VISIBLE GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus


def scoring(sess, table, cond_A, fake_B, graph):
    """ collect result
    inputs
    ------
    sess : 
    table : dict
    cond_A : 
    fake_B : ß

    """
    # placeholder tensor
    G_samples_t = graph.get_tensor_by_name('G_samples:0')
    matched_cond_t = graph.get_tensor_by_name('matched_cond:0')
    critic_scores_t = graph.get_tensor_by_name(
        'Critic/C_inference_1/linear_result/BiasAdd:0')
    # 'Generator/G_loss/C_inference/linear_result/Reshape:0')
    critic_scores_all = []
    if cond_A.shape[1] == 256:
        cond_A = cond_A.reshape([-1, 128, 100, 13])
        fake_B = fake_B.reshape([-1, 128, 100, 10])
    for batch_id in range(cond_A.shape[0]):
        feed_dict = {
            G_samples_t: fake_B[batch_id],
            matched_cond_t: cond_A[batch_id]
        }
        critic_scores = sess.run(
            critic_scores_t, feed_dict=feed_dict)
        critic_scores_all.append(critic_scores)
    critic_scores_all = np.array(critic_scores_all).flatten()
    critic_scores_sorted = np.sort(critic_scores_all)
    table['average'] = str(np.mean(critic_scores_sorted))
    table['average_top_10'] = str(np.mean(critic_scores_sorted[-10:]))
    table['average_top_100'] = str(np.mean(critic_scores_sorted[-100:]))


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # sesstion config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
        with tf.Session(config=config) as sess:
            # restored
            saver.restore(sess, FLAGS.restore_path)
            score_table = {}
            for path, dirs, files in os.walk(FLAGS.folder_path):
                if 'collect' in path:
                    cond_A = None
                    fake_B = None
                    for file_name in files:
                        if 'results_A' in file_name:
                            cond_A = np.load(os.path.join(path, file_name))
                        if 'results_fake_B' in file_name:
                            fake_B = np.load(os.path.join(path, file_name))
                    if cond_A is not None and fake_B is not None:
                        score_table[path] = {}
                        with open(os.path.join(path, '../hyper_parameters.json')) as hyper_json:
                            score_table[path]['comment'] = json.load(hyper_json)[
                                'comment']
                        scoring(sess, score_table[path],
                                cond_A, fake_B, graph=graph)
                        print("finish %s !!" % (path))
            # dump result
            with open(SCORE_TABLE_PATH, 'w') as outfile:
                json.dump(score_table, outfile)


if __name__ == '__main__':
    assert FLAGS.restore_path is not None
    tf.app.run()
