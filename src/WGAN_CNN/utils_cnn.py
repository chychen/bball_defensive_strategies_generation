"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Norm(object):
    """ 
    * singletance pattern
    * discrete player position onto sparse map (shape=[100,50])
    """
    __instance = None
    COLS = 96
    ROWS = 48
    PLAYERS = 11
    BASKET_LEFT = [4, 25]
    BASKET_RIGHT = [90, 25]

    def __new__(clz, real_data=None):
        if not Norm.__instance:
            Norm.__instance = object.__new__(clz)
        else:
            print("Instance Exists! :D")
        return Norm.__instance

    def __init__(self, real_data=None):
        """ TODO normalize output?!
        """
        if real_data is not None:
            self.max_x = np.amax(real_data[:, :, :, 0])
            self.min_x = np.amin(real_data[:, :, :, 0])
            self.max_y = np.amax(real_data[:, :, :, 1])
            self.min_y = np.amin(real_data[:, :, :, 1])
            # + 1e-4 to make sure 0-based coordinate
            self.range_x = self.max_x - self.min_x + 1e-4
            self.range_y = self.max_y - self.min_y + 1e-4
            print('self.range_x:', self.range_x)
            print('self.range_y:', self.range_y)

    def map_2_position(self, map_):
        """ 
        Args
        ----
        map_ : float, shape=[batch_size, length=100, cols=100, rows=50, players=11]

        Returns
        -------
        result : float, shape=[batch_size, length=100, features=23]

        Note
        ----
        features : 1 ball*3 (xyz) + 10 players*2 (xy)

        """
        shape_ = map_.shape
        result = []
        ball_z = np.zeros(shape=[shape_[0], shape_[1]])
        # ball
        ball_on_map = map_[:, :, :, :, 0]
        ball_position = np.argmax(ball_on_map.reshape(
            [shape_[0], shape_[1], -1]), axis=-1)
        result.append(ball_position // Norm.ROWS)
        result.append(ball_position % Norm.ROWS)
        result.append(ball_z)
        # players
        for player in range(1, Norm.PLAYERS):
            player_on_map = map_[:, :, :, :, player]
            player_position = np.argmax(player_on_map.reshape(
                [shape_[0], shape_[1], -1]), axis=-1)
            result.append(player_position // Norm.ROWS)
            result.append(player_position % Norm.ROWS)
        result = np.stack(result, axis=-1)
        print(result.shape)
        return result

    def format_discrete_map(self, batch_data):
        """ TODO z and player position
        Args
        ----
        batch_data : float, shape=[batch_size, length=100, players=11, features=4]

        Returns
        -------
        map_ : float, shape=[batch_size, length=100, players=11, cols=100, rows=50]

        """
        shape_ = batch_data.shape
        coor_x = (batch_data[:, :, :, 0] - self.min_x) * \
            Norm.COLS // self.range_x
        coor_y = (batch_data[:, :, :, 1] - self.min_y) * \
            Norm.ROWS // self.range_y

        map_ = np.zeros(shape=[shape_[0], shape_[1], shape_[
                        2], Norm.COLS, Norm.ROWS], dtype=np.float32)
        coor_x = coor_x.reshape([-1]).astype(np.int32)
        coor_y = coor_y.reshape([-1]).astype(np.int32)
        map_ = map_.reshape([-1, Norm.COLS, Norm.ROWS])
        idx_ = [x for x in range(map_.shape[0])]
        map_[idx_, coor_x, coor_y] = 1.0
        map_ = map_.reshape([shape_[0], shape_[1], Norm.PLAYERS,
                             Norm.COLS, Norm.ROWS])
        return map_


def testing():
    # dummy = np.ones(shape=[32, 100, 11, 4])
    dummy = np.load('../../data/F2.npy')[:32]
    print(dummy.shape)
    print('#1')
    normer = Norm(dummy)

    print('#2')
    result = normer.format_discrete_map(dummy)
    print(result.shape)
    print(result.dtype)

    dummy = np.ones(shape=[32, 100, 100, 50, 11])
    print('#3')
    result = normer.map_2_position(dummy)


if __name__ == '__main__':
    testing()
