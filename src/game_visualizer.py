import numpy as np
import argparse

import os
from os import listdir
from os.path import join

import time
import matplotlib
matplotlib.use('agg')  # run backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc

from utils import Norm

# * B <-> 0
# * F <-> 1
# * G <-> 2
# * C-F <-> 3
# * F-G <-> 4
# * F-C <-> 5
# * C <-> 6
# * G-F <-> 7
PP_LIST = ['B', 'F', 'G', 'C-F', 'F-G', 'F-C', 'C', 'G-F']


def update_all(frame_id, player_circles, ball_circle, annotations, data):
    """ 
    Inputs
    ------
    frame_id : int
        automatically increased by 1
    player_circles : list of pyplot.Circle
        players' icon
    ball_circle : list of pyplot.Circle
        ball's icon
    annotations : pyplot.axes.annotate
        colors, texts, locations for ball and players
    data : float, shape=[amount, length, 23]
        23 = ball's xyz + 10 players's xy
    """
    max_length = data.shape[1]
    # players
    for j, circle in enumerate(player_circles):
        circle.center = data[frame_id // max_length, frame_id %
                             max_length, 3 + j * 2 + 0], data[frame_id // max_length, frame_id % max_length, 3 + j * 2 + 1]
        annotations[j].set_position(circle.center)
    # print("Frame:", frame_id)
    # ball
    ball_circle.center = data[frame_id // max_length, frame_id %
                              max_length, 0], data[frame_id // max_length, frame_id % max_length, 1]
    ball_circle.set_radius(1.0 + data[frame_id // max_length, frame_id % max_length, 2] / 10.0)
    annotations[10].set_position(ball_circle.center)
    return


def plot_data(data, length, file_path=None, if_save=False, fps=4, dpi=48):
    """
    Inputs
    ------
    data : float, shape=[amount, length, 23 + 70]
        93 = ball's xyz + 10 players's xy + 10 * 7-dims-one-hot 
    length : int
        how long would you like to plot
    file_path : str
        where to save the animation
    if_save : bool, optional
        save as .gif file or not
    fps : int, optional
        frame per second
    dpi : int, optional
        dot per inch
    Return
    ------
    """
    court = plt.imread("../data/court.png")  # 500*939
    # get ten 7-dims-one-hot of player positions
    onehot_vec = data[0, 0, 23:].reshape([10, 7])
    players_list = np.argmax(onehot_vec, axis=-1) + 1  # 0 <-> Ball
    ball_value = np.zeros(shape=[1])
    name_scalar_list = np.concatenate([players_list, ball_value], axis=-1)
    print(name_scalar_list.shape)
    name_list = []
    for v in name_scalar_list:
        name_list.append(PP_LIST[int(v)])

    # team A -> read circle, team B -> blue circle, ball -> small green circle
    player_circles = []
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='r'))
     for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=2.5, color='b'))
     for _ in range(5)]
    ball_circle = plt.Circle(xy=(0, 0), radius=1.5, color='g')

    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(0, 50))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    # annotations on circles
    annotations = [ax.annotate(name_list[i], xy=[0, 0],
                               horizontalalignment='center',
                               verticalalignment='center', fontweight='bold')
                   for i in range(11)]
    # animation
    anim = animation.FuncAnimation(fig, update_all, fargs=(
        player_circles, ball_circle, annotations, data), frames=length, interval=100)

    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    if if_save:
        anim.save(file_path, fps=fps,
                  dpi=dpi, writer='imagemagick')
        print('!!!gif animation is saved!!!')
    else:
        plt.show()
        print('!!!End!!!')

    # clear content
    plt.cla()
    plt.clf()


def test():
    """
    test only
    """
    train_data = np.load(opt.data_path)
    normer = Norm(train_data)
    train_data = normer.get_normed_data()
    train_data = normer.recover_data(train_data)

    plot_data(train_data, length=opt.length,
              file_path=opt.save_path, if_save=opt.save)
    print('opt.save', opt.save)
    print('opt.length', opt.length)
    print('opt.seq_length', opt.seq_length)
    print('opt.save_path', opt.save_path)


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser(description='NBA Games visulization')
    parser.add_argument('--save', type=bool, default=True,
                        help='bool, if save as gif file')
    parser.add_argument('--length', type=int, default=100,
                        help='how many frames do you want to plot')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='how long for each event')
    parser.add_argument('--save_path', type=str, default='../data/ten_event.gif',
                        help='string, path to save event animation')
    parser.add_argument('--data_path', type=str,
                        default='../data/FEATURES.npy', help='string, path of target data')

    opt = parser.parse_args()
    test()
