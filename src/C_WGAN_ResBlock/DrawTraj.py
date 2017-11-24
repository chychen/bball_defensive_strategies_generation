import numpy as np
import matplotlib.pyplot as plt
import os

#data path
data_path = "./Data/results_A_fake_B.npy"
#save image file path
save_path = './Data/Images/'
file_name = 'FakeData'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

#parameters
data_TF = False   #real data or generated
alpha_ = 0       #alpha
start = 0   #starting point
end = 19     #ending point (20 frames per image)
count = 0
#load data
data = np.load(data_path)

if data_TF == True:
    p1x = data[0,:,0]
    p1y = data[0,:,1]
    p2x = data[0,:,3]
    p2y = data[0,:,4]
    p3x = data[0,:,5]
    p3y = data[0,:,6]
    p4x = data[0,:,7]
    p4y = data[0,:,8]
    p5x = data[0,:,9]
    p5y = data[0,:,10]
    p6x = data[0,:,11]
    p6y = data[0,:,12]
    p7x = data[0,:,13]
    p7y = data[0,:,14]
    p8x = data[0,:,15]
    p8y = data[0,:,16]
    p9x = data[0,:,17]
    p9y = data[0,:,18]
    p10x = data[0,:,19]
    p10y = data[0,:,20]
    p11x = data[0,:,21]
    p11y = data[0,:,22]
if data_TF == False:
    p1x = data[0,0,:,0]
    p1y = data[0,0,:,1]
    p2x = data[0,0,:,3]
    p2y = data[0,0,:,4]
    p3x = data[0,0,:,5]
    p3y = data[0,0,:,6]
    p4x = data[0,0,:,7]
    p4y = data[0,0,:,8]
    p5x = data[0,0,:,9]
    p5y = data[0,0,:,10]
    p6x = data[0,0,:,11]
    p6y = data[0,0,:,12]
    p7x = data[0,0,:,13]
    p7y = data[0,0,:,14]
    p8x = data[0,0,:,15]
    p8y = data[0,0,:,16]
    p9x = data[0,0,:,17]
    p9y = data[0,0,:,18]
    p10x = data[0,0,:,19]
    p10y = data[0,0,:,20]
    p11x = data[0,0,:,21]
    p11y = data[0,0,:,22]

fig, ax = plt.subplots()
court = plt.imread("fullcourt.png")

for i in range(5):
    for x in range(start, end):
        #alpha higher as timestep increases
        alpha_ = 0.01 * count
        #ball trajectory
        ax.plot(p1x[x:end], p1y[x:end], c='g', alpha=alpha_, linewidth=4, solid_capstyle='round')
        #offensive player trajectory * 5
        ax.plot(p2x[x:end], p2y[x:end], c='r', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p3x[x:end], p3y[x:end], c='r', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p4x[x:end], p4y[x:end], c='r', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p5x[x:end], p5y[x:end], c='r', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p6x[x:end], p6y[x:end], c='r', alpha=alpha_, linewidth=4, solid_capstyle='round')
        #defensive player trajecotry * 5
        ax.plot(p7x[x:end], p7y[x:end], c='b', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p8x[x:end], p8y[x:end], c='b', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p9x[x:end], p9y[x:end], c='b', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p10x[x:end], p10y[x:end], c='b', alpha=alpha_, linewidth=4, solid_capstyle='round')
        ax.plot(p11x[x:end], p11y[x:end], c='b', alpha=alpha_, linewidth=4, solid_capstyle='round')

        count+=1

    #trajectory label
    ax.annotate('ball', (p1x[end], p1y[end]))
    ax.annotate('A1', (p2x[end], p2y[end]))
    ax.annotate('A2', (p3x[end], p3y[end]))
    ax.annotate('A3', (p4x[end], p4y[end]))
    ax.annotate('A4', (p5x[end], p5y[end]))
    ax.annotate('A5', (p6x[end], p6y[end]))
    ax.annotate('B1', (p7x[end], p7y[end]))
    ax.annotate('B2', (p8x[end], p8y[end]))
    ax.annotate('B3', (p9x[end], p9y[end]))
    ax.annotate('B4', (p10x[end], p10y[end]))
    ax.annotate('B5', (p11x[end], p11y[end]))

    plt.axis('off')
    plt.imshow(court, zorder=0, extent=[0, 94, 50, 0])
    #plt.show()
    plt.savefig(save_path+file_name+'{}.png'.format(i))
    ax.cla()
    start += 20
    end += 20
    alpha_ = 0
    count = 0

print("Finished")