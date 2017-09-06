import numpy as np
import argparse

import os
from os import listdir
from os.path import join
from PIL import Image
# from tensorboard import SummaryWriter

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc



parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--path', default='')
parser.add_argument('--Ex', default=90)
parser.add_argument('--RandomSeed', type=int, default=1, help='Random seed')


opt = parser.parse_args()


# path = "../store/" + opt.path + "/G_E" + str(opt.Ex) +".pth"
# G.load_state_dict(torch.load(path))
train_data = np.load('unitTest.npy')
train_data = train_data[:1000,:]
def update_all(i, player_circles, ball_circle, annotations, oh):
    for j, circle in enumerate(player_circles):
        # circle.center = np.random.randint(0, 70), np.random.randint(0, 48)
        circle.center = oh[0, i , j * 3 +3], oh[0, i , j * 3 +4]
        annotations[j].set_position(circle.center)
    print("Frame", i)
    ball_circle.center = oh[0, i , 0], oh[0, i, 1]
    annotations[10].set_position(ball_circle.center)
    return player_circles, ball_circle



def GoPredict(data):

    go = data

    # one event
    oh = go[0, :, :]
    name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'Ball']
    color = ['red','red','red','red','red','blue','blue','blue','blue','blue','blue']
    ax = plt.axes(xlim=(0, 100), ylim=(0,50))

    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)

    player_circles = [plt.Circle((0,0), 16/7) for i in range(10)]

    ball_circle = plt.Circle((0,0), 12/7)

    for circle in player_circles:
        ax.add_patch(circle)
    ax.add_patch(ball_circle)

    annotations = [ax.annotate(name[i], xy=[0, 0], color=color[i],
                    horizontalalignment='center',
                    verticalalignment='center', fontweight='bold')
                    for i in range(11)]

    anim = animation.FuncAnimation(fig, update_all, fargs=(player_circles, ball_circle, annotations, go), frames=1000, interval=100)
    
    court = plt.imread("court.png")
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    #if save:
    # anim.save('Match.gif', dpi=80, writer='imagemagick')
    plt.show()
    print(go.shape)

GoPredict(train_data)