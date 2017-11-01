"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Norm(object):
    """singletance pattern
    """
    __instance = None

    def __new__(clz, real_data=None):
        if not Norm.__instance:
            Norm.__instance = object.__new__(clz)
        else:
            print("Instance Exists! :D")
        return Norm.__instance

    def __init__(self, real_data=None):
        """
        params
        ------
        real_data : float, shape=[#, length=100, players=11, features=4]

        note
        ----
        feature :
            x, y, z, and player position
        """
        if real_data is not None:
            self.__real_data = real_data
            self.__basket_left = [4, 25]
            self.__basket_right = [90, 25]
            self.__norm_dict = {}
            # position normalization
            self.__normalize_pos()
            # player position encoding
            self.__encode_10_onehot()

    def get_normed_data(self):
        return self.__real_data

    def recover_data(self, norm_data):
        # X
        norm_data[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = norm_data[:, :, [
            0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        norm_data[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = norm_data[:, :, [
            1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        # Z
        norm_data[:, :, 2] = norm_data[:, :, 2] * \
            self.__norm_dict['z']['stddev'] + self.__norm_dict['z']['mean']
        return norm_data

    def extract_features(self, t_data):
        """ extract 272 features from raw data, including
        * x y z positions = 23
        * x y z speed = 23, first frame's speed should be zero
        * x y correlation of 10 players, ball, and 2 basket = (13*13-13)/2*2=156
        * 10 one-hot vector as player position = 70

        params
        ------
        t_data : tensor, float, shape=[batch, length, features=23+70]
            the sequence data
        note
        ----
        features : 23+70
            23 = 10 players + ball
            70 = 10 7dims-one-hot vector as player position
        """
        with tf.name_scope('extract') as scope:
            # info
            t_shape = t_data.get_shape().as_list()

            # x y z positions = 23
            t_pos = t_data[:, :, :23]
            # x y z speed = 23
            t_speed = tf.concat(
                [(t_pos[:, 1:t_shape[1], :] - t_pos[:, 0:t_shape[1]-1, :]), tf.zeros(shape=[t_shape[0], 1, 23])], axis=1)
            # x y correlation of 1 ball, 10 players and 2 basket = (13*13-13)/2*2=156
            t_correlation = []
            for axis_ in range(2):
                t_basket_l = tf.ones(
                    shape=[t_shape[0], t_shape[1], 1]) * self.__basket_left[axis_]
                t_basket_r = tf.ones(
                    shape=[t_shape[0], t_shape[1], 1]) * self.__basket_right[axis_]
                t_x = tf.concat([t_pos[:, :, 0 + axis_:1 + axis_],
                                 t_pos[:, :, 3 + axis_::2], t_basket_l, t_basket_r], axis=2)
                for i in range(13):
                    for j in range(1, i + 1):
                        t_vec = t_x[:, :, i] - t_x[:, :, j]
                        t_correlation.append(t_vec)
            t_correlation = tf.stack(t_correlation, axis=-1)
            # 10 one-hot vector as player position = 70
            t_onehot = t_data[:, :, 23:23 + 70]
            return tf.concat([t_pos, t_speed, t_correlation, t_onehot], axis=-1)

    def __normalize_pos(self):
        """ directly normalize player x,y,z on self.__real_data
        """
        axis_list = ['x', 'y', 'z']
        for i, axis_ in enumerate(axis_list):
            if axis_ == 'z':  # z
                mean_ = np.mean(
                    self.__real_data[:, :, 0, i])
                stddev_ = np.std(
                    self.__real_data[:, :, 0, i])
                self.__real_data[:, :, 0, i] = (
                    self.__real_data[:, :, 0, i] - mean_) / stddev_
                self.__norm_dict[axis_] = {}
                self.__norm_dict[axis_]['mean'] = mean_
                self.__norm_dict[axis_]['stddev'] = stddev_
            else:  # x and y
                mean_ = np.mean(
                    self.__real_data[:, :, :, i])
                stddev_ = np.std(
                    self.__real_data[:, :, :, i])
                self.__real_data[:, :, :, i] = (
                    self.__real_data[:, :, :, i] - mean_) / stddev_
                self.__basket_left[i] = (
                    self.__basket_left[i] - mean_) / stddev_
                self.__basket_right[i] = (
                    self.__basket_right[i] - mean_) / stddev_
                self.__norm_dict[axis_] = {}
                self.__norm_dict[axis_]['mean'] = mean_
                self.__norm_dict[axis_]['stddev'] = stddev_

    def __encode_10_onehot(self):
        """ directly add player positions one-hot vec on self.__real_data

        note
        ----
        player position :
            * BALL <-> 0
            * F <-> 1
            * G <-> 2
            * C-F <-> 3
            * F-G <-> 4
            * F-C <-> 5
            * C <-> 6
            * G-F <-> 7
        """
        player_position = self.__real_data[:, :,
                                           1:, -1].astype(np.int)  # without ball
        ten_onehot = np.zeros(
            shape=[player_position.shape[0], player_position.shape[1], 10, 7])
        for i in range(player_position.shape[0]):
            for j in range(player_position.shape[1]):
                for k in range(player_position.shape[2]):
                    ten_onehot[i, j, k, player_position[i][j][k] - 1] = 1
        # print(self.__real_data[10, 10, :, 3])
        # print(ten_onehot[10, 10, :, :])
        self.__real_data = np.concatenate(
            [
                # ball
                self.__real_data[:, :, 0, :3].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3]),
                # players
                self.__real_data[:, :, 1:, :2].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 10 * 2]),
                # player position as ten 7dims-one-hot vector
                ten_onehot.reshape(
                    [player_position.shape[0], player_position.shape[1], 10 * 7])
            ], axis=-1
        )

def testing_real():
    real_data = np.load("../data/FEATURES.npy")
    print('real_data.shape', real_data.shape)
    normer = Norm(real_data)
    real_data = normer.get_normed_data()
    print('real_data.shape', real_data.shape)

def testing():
    dummy = np.ones(shape=[512, 100, 11, 4])
    normer = Norm(dummy)
    dummy_samples = tf.ones(shape=[512, 100, 23 + 70])
    normer.extract_features(dummy_samples)


if __name__ == '__main__':
    testing_real()
