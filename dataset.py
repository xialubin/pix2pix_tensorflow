import numpy as np
from skimage import io, color
from queue import Queue
from threading import Thread
import random
import sys


class Dateset(object):
    def __init__(self, batch_size, image_size, data_path):
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_path = data_path

        self.record_queue = Queue(maxsize=1000)
        self.batch_queue = Queue(maxsize=10)

        self.record_list = []
        files = open(data_path, 'r')
        for file_name in files:
            file_name = file_name.strip()
            self.record_list.append(file_name)

        self.record_number = len(self.record_list)
        self.record_point = 0
        self.batch_num = self.record_number // batch_size

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(5):
            t = Thread(target=self.batch_producer)
            t.daemon = True
            t.start()

    def record_producer(self):
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def batch_producer(self):
        while True:
            batch_rgb = []
            for i in range(self.batch_size):
                img_name = self.record_queue.get()
                img_rgb = io.imread(img_name)
                batch_rgb.append(img_rgb)
            batch_lab = color.rgb2lab(np.array(batch_rgb))
            batch_l = batch_lab[:, :, :, 0:1] / 50.0 - 1.0
            batch_ab = batch_lab[:, :, :, 1:] / 110.0
            batch = {'l': batch_l, 'ab': batch_ab}
            self.batch_queue.put(batch)

    def get_batch_list(self):
        random.shuffle(self.record_list)
        batch_list = []
        for i in range(self.batch_num):
            batch = self.record_list[i:(i + self.batch_size)]
            batch_list.append(batch)
        return batch_list

    def get_image(self, img_name_list):
        batch_rgb = []
        for i in range(self.batch_size):
            img_rgb = io.imread(img_name_list[i])
            batch_rgb.append(img_rgb)
        batch_rgb = np.array(batch_rgb)
        batch_lab = color.rgb2lab(batch_rgb)
        batch_l = batch_lab[:, :, :, 0:1] / 50.0 - 1.0
        batch_ab = batch_lab[:, :, :, 1:] / 110.0
        batch = {'l': batch_l, 'ab': batch_ab}
        return batch

    def get_batch(self):
        return self.batch_queue.get()
