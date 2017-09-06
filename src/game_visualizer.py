import numpy as np
import argparse

import os
from os import listdir
from os.path import join
# from tensorboard import SummaryWriter

import time
import matplotlib
matplotlib.use('agg') # run backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc


parser = argparse.ArgumentParser(description='NBA Games visulization')
parser.add_argument('--save', type=bool, default=False,
                    help='bool, if save as gif file')
parser.add_argument('--length', type=int, default=300,
                    help='how many frames do you want to plot')
parser.add_argument('--save_path', type=str, default='../data/ten_event.gif',
                    help='string, path to save event animation')
parser.add_argument('--data_path', type=str,
                    default='../data/NBA-TEAM1.npy', help='string, path of target data')


opt = parser.parse_args()


def update_all(frame_id, player_circles, ball_circle, annotations, data, max_length=300):
    """
    TODO
    """
    # players
    for j, circle in enumerate(player_circles):
        circle.center = data[frame_id // max_length, frame_id %
                             max_length, 3 + j * 2 + 0], data[frame_id // max_length, frame_id % max_length, 3 + j * 2 + 1]
        annotations[j].set_position(circle.center)
    # print("Frame:", frame_id)
    # ball
    ball_circle.center = data[frame_id // max_length, frame_id %
                              max_length, 0], data[frame_id // max_length, frame_id % max_length, 1]
    annotations[10].set_position(ball_circle.center)
    return player_circles, ball_circle


def plot_data(data, length, file_path=None, if_save=False, fps=96, dpi=48):
    """
    TODO
    """
    court = plt.imread("../data/court.png")  # 500*939

    # 5 A-Team players + 5 B-Team players + 1 ball
    name_list = ['A1', 'A2', 'A3', 'A4',
                 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', ' ']
    # color = ['red', 'red', 'red', 'red', 'red', 'blue',
    #          'blue', 'blue', 'blue', 'blue', 'yellow']

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
        
    plt.cla()
    plt.clf()

def main():
    # load data and remove useless z dimension of players in data
    train_data = np.load(opt.data_path)[:, :, [
        0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31]]
    plot_data(train_data, length=opt.length,
              file_path=opt.save_path, if_save=opt.save)
    print(opt.length)
    print(opt.save_path)
    print(opt.save)


if __name__ == '__main__':
    main()
