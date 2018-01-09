from subprocess import Popen, PIPE
import os

if not os.path.exists('ckpt'):
    process = Popen(['wget', '-P', './', 'http://140.113.210.14:6006//NBA/data/ckpt.zip'], stdout=PIPE, stderr=PIPE)
    print('start Download')
    stdout, stderr = process.communicate()
    print('Finish download!')
    process = Popen(['unzip', './ckpt.zip', '-d', './' ], stdout=PIPE, stderr=PIPE)
    print('start unzip')
    stdout, stderr = process.communicate()
    print('Finish unzip!')
if not os.path.exists('../data/FEATURES-4.npy'):
    process = Popen(['wget', '-P', '../data', 'http://140.113.210.14:6006//NBA/data/FEATURES-4.npy'], stdout=PIPE, stderr=PIPE)
    print('start Download')
    stdout, stderr = process.communicate()
    print('Finish download!')