import os

path = 'D:/XiaLubin/project/python/pix2pix/data/colorize/test/'
train = open('./test.txt', 'w')
for name in os.listdir(path):
    train.write(path + name)
    train.write('\n')
train.close()