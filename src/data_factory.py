"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils import Norm


class DataSource(object):
    """
    """
    def __init__(self, filepath):
        # load data and remove useless z dimension of players in data
        real_data = np.load(filepath)
        print('real_data.shape', real_data.shape)

        # normalization
        normer = Norm(real_data)
        real_data = normer.get_normed_data()[:, :FLAGS.seq_length, :]
        print(real_data.shape)

        # number of batches
        num_batches = real_data.shape[0] // FLAGS.batch_size
