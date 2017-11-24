"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class DataFactory(object):
    """singletance pattern
    """
    __instance = None

    def __new__(clz, real_data=None):
        if not DataFactory.__instance:
            DataFactory.__instance = object.__new__(clz)
        else:
            print("Instance Exists! :D")
        return DataFactory.__instance

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
            # position normalization
            self.__norm_dict = self.__normalize_pos()
            # make training data ready
            self.train_data, self.valid_data = self.__get_ready()

    def fetch_data(self):
        return self.train_data, self.valid_data

    def fetch_ori_data(self):
        return np.concatenate(
            [
                # ball
                self.__real_data[:, :, 0, :3].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3]),
                # team A players
                self.__real_data[:, :, 1:, :2].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 10 * 2])
            ], axis=-1
        )

    def extract_features(self, t_data):
        """ extract 202 features from raw data, including
        * x y z positions = 23
        * x y z speed = 23, first frame's speed should be zero
        * x y correlation of 10 players, ball, and 2 basket = (13*13-13)/2*2=156

        params
        ------
        t_data : tensor, float, shape=[batch, length, features=23]
            the sequence data
        note
        ----
        features : 23
            23 = 10 players + ball
        """
        with tf.name_scope('extract') as scope:
            # info
            t_shape = t_data.get_shape().as_list()

            # x y z positions = 23
            t_pos = t_data[:, :, :23]
            # x y z speed = 23
            t_speed = tf.concat(
                [(t_pos[:, 1:t_shape[1], :] - t_pos[:, 0:t_shape[1] - 1, :]), tf.zeros(shape=[t_shape[0], 1, 23])], axis=1)
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
            return tf.concat([t_pos, t_speed, t_correlation], axis=-1)

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

    def shuffle(self):
        shuffled_indexes = np.random.permutation(self.train_data['A'].shape[0])
        self.train_data['A'] = self.train_data['A'][shuffled_indexes]
        self.train_data['B'] = self.train_data['B'][shuffled_indexes]
        shuffled_indexes = np.random.permutation(self.valid_data['A'].shape[0])
        self.valid_data['A'] = self.valid_data['A'][shuffled_indexes]
        self.valid_data['B'] = self.valid_data['B'][shuffled_indexes]
        return self.train_data, self.valid_data

    def __get_ready(self):
        train = {}
        valid = {}
        shuffled_indexes = np.random.permutation(self.__real_data.shape[0])
        # A
        team_A = np.concatenate(
            [
                # ball
                self.__real_data[:, :, 0, :3].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 1 * 3]),
                # team A players
                self.__real_data[:, :, 1:6, :2].reshape(
                    [self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2])
            ], axis=-1
        )[shuffled_indexes]
        train['A'], valid['A'] = np.split(
            team_A, [self.__real_data.shape[0] // 10 * 9])
        print(train['A'].shape)
        print(valid['A'].shape)
        # B
        team_B = self.__real_data[:, :, 6:11, :2].reshape(
            [self.__real_data.shape[0], self.__real_data.shape[1], 5 * 2]
        )[shuffled_indexes]
        train['B'], valid['B'] = np.split(
            team_B, [self.__real_data.shape[0] // 10 * 9])
        print(train['B'].shape)
        print(valid['B'].shape)
        return train, valid

    def __normalize_pos(self):
        """ directly normalize player x,y,z on self.__real_data
        """
        norm_dict = {}
        axis_list = ['x', 'y', 'z']
        for i, axis_ in enumerate(axis_list):
            if axis_ == 'z':  # z
                mean_ = np.mean(
                    self.__real_data[:, :, 0, i])
                stddev_ = np.std(
                    self.__real_data[:, :, 0, i])
                self.__real_data[:, :, 0, i] = (
                    self.__real_data[:, :, 0, i] - mean_) / stddev_
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
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
                norm_dict[axis_] = {}
                norm_dict[axis_]['mean'] = mean_
                norm_dict[axis_]['stddev'] = stddev_
        return norm_dict


def testing_real():
    pass
    # real_data = np.load("../data/FEATURES.npy")
    # print('real_data.shape', real_data.shape)
    # data_ = DataFactory(real_data)
    # real_data = normer.get_normed_data()
    # print('real_data.shape', real_data.shape)


if __name__ == '__main__':
    testing_real()
