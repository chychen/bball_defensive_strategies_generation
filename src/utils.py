"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Norm(object):
    __instance = None

    def __new__(clz, real_data=None):
        if not Norm.__instance:
            Norm.__instance = object.__new__(clz)
            print("__new__")
        return Norm.__instance

    def __init__(self, real_data=None):
        print("__init__")
        if real_data is not None:
            self.__real_data = real_data
            self.__basket_left = [4, 25]
            self.__basket_right = [90, 25]
            self.__norm_dict = {}
            # X
            mean_x = np.mean(
                real_data[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
            stddev_x = np.std(
                real_data[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
            real_data[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = (
                real_data[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] - mean_x) / stddev_x
            self.__basket_left[0] = (self.__basket_left[0] - mean_x) / stddev_x
            self.__basket_right[0] = (self.__basket_right[0] - mean_x) / stddev_x
            self.__norm_dict['x'] = {}
            self.__norm_dict['x']['mean'] = mean_x
            self.__norm_dict['x']['stddev'] = stddev_x
            # Y
            mean_y = np.mean(
                real_data[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]])
            stddev_y = np.std(
                real_data[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]])
            real_data[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = (
                real_data[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] - mean_y) / stddev_y
            self.__basket_left[1] = (self.__basket_left[1] - mean_y) / stddev_y
            self.__basket_right[1] = (self.__basket_right[1] - mean_y) / stddev_y
            self.__norm_dict['y'] = {}
            self.__norm_dict['y']['mean'] = mean_y
            self.__norm_dict['y']['stddev'] = stddev_y
            # Z
            mean_z = np.mean(
                real_data[:, :, 2])
            stddev_z = np.std(
                real_data[:, :, 2])
            real_data[:, :, 2] = (
                real_data[:, :, 2] - mean_z) / stddev_z
            self.__norm_dict['z'] = {}
            self.__norm_dict['z']['mean'] = mean_z
            self.__norm_dict['z']['stddev'] = stddev_z

    def get_normed_data(self):
        return self.__real_data

    def recover_data(self, norm_data):
        # X
        samples[:, :, [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] = samples[:, :, [
            0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]] * self.__norm_dict['x']['stddev'] + self.__norm_dict['x']['mean']
        # Y
        samples[:, :, [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] = samples[:, :, [
            1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]] * self.__norm_dict['y']['stddev'] + self.__norm_dict['y']['mean']
        # Z
        samples[:, :, 2] = samples[:, :, 2] * \
            self.__norm_dict['z']['stddev'] + self.__norm_dict['z']['mean']
        return samples

    def extract_features(t_data):
        """ extract 252 features from raw data, including 
        * x y z positions = 23
        * x y z speed = 23, first frame's speed should be zero
        * x y correlation of 10 players, ball, and 2 basket = (13*13-13)/2*2=156
        * 10 one-hot vector as player position = 50

        params
        ------
        t_data : tensor, float, shape=[batch, length, features=23+50]
            the sequence data
        note
        ----
        features : 23+50
            23 = 10 players + ball
            50 = 10 one-hot vector as player position
        """
        # info
        t_shape = t_pos.get_shape().as_list()

        # x y z positions = 23
        t_pos = t_data[:, :, :23]
        # x y z speed = 23
        t_speed = tf.stack(
            [(t_pos[:, 1:100, :] - t_pos[:, 0:99, :]), tf.zeros(shape=[t_shape[0], 1, 23])], axis=1)
        # x y correlation of 1 ball, 10 players and 2 basket = (13*13-13)/2*2=156
        t_correlation = []
        t_basket_l = tf.ones(shape=[t_shape[0], t_shape[1], 1]) * self.__basket_left[0]
        t_basket_r = tf.ones(shape=[t_shape[0], t_shape[1], 1]) * self.__basket_right[0]
        t_x = tf.stack([t_pos[:, :, 0], t_pos[:, :, 3::2], t_basket_l, t_basket_r], axis=-1)
        t_basket_l = tf.ones(shape=[t_shape[0], t_shape[1], 1]) * self.__basket_left[1]
        t_basket_r = tf.ones(shape=[t_shape[0], t_shape[1], 1]) * self.__basket_right[1]
        t_y = tf.stack([t_pos[:, :, 1], t_pos[:, :, 4::2], t_basket_l, t_basket_r], axis=-1)

        # 10 one-hot vector as player position = 50


if __name__ == '__main__':
    Norm()
    Norm()
