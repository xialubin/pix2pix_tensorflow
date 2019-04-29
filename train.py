from network import Generator_Unet, Discriminator
import tensorflow as tf
import sys
from dataset import Dateset
import numpy as np
from skimage import color, io

import warnings
warnings.filterwarnings("ignore")


class Model(object):
    def __init__(self, encoder_kernels, decoder_kernels, dis_kernels, trainning=True):
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.dis_kernels = dis_kernels
        self.training = trainning
        self.gen_factor = Generator_Unet(name='gen', encoder_kernels=self.encoder_kernels,
                                         decoder_kernels=self.decoder_kernels)
        self.dis_factor = Discriminator(name='dis', kernels=self.dis_kernels)
        self.batch_size = 4
        self.image_size = 256
        self.train_data = Dateset(batch_size=self.batch_size, image_size=self.image_size, data_path='./data/colorize/train.txt')
        self.test_data = Dateset(batch_size=self.batch_size * 2, image_size=self.image_size, data_path='./data/colorize/test.txt')

    def net_initial(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./logs/logs_ab2gray_pix2pixloss/', self.sess.graph)

    def build(self, lambda_pixel, learning_rate=0.0002):
        self.input_gray = tf.placeholder(shape=[None, 256, 256, 1], dtype=tf.float32, name='input_gray')
        self.input_ab = tf.placeholder(shape=[None, 256, 256, 2], dtype=tf.float32, name='input_ab')
        gen_output = self.gen_factor.create(self.input_ab)  # 这个地方输入为ab，输出为gray了
        dis_input_fake = tf.concat([gen_output, self.input_ab], axis=-1)  # concat的时候也要改一下顺序
        dis_output_fake = self.dis_factor.create(dis_input_fake)
        dis_input_real = tf.concat([self.input_gray, self.input_ab], axis=-1)
        dis_output_real = self.dis_factor.create(dis_input_real, reuse_variable=True)

        # discriminator loss
        loss_dis_fake = tf.losses.mean_squared_error(tf.zeros_like(dis_output_fake), dis_output_fake,
                                                     scope='loss_dis_fake')
        loss_dis_real = tf.losses.mean_squared_error(tf.ones_like(dis_output_real), dis_output_real,
                                                     scope='loss_dis_real')
        self.loss_D = (loss_dis_fake + loss_dis_real) / 2.0
        tf.summary.scalar('loss_dis_fake', loss_dis_fake)
        tf.summary.scalar('loss_dis_real', loss_dis_real)
        tf.summary.scalar('loss_D', self.loss_D)

        # generator loss
        loss_gen = tf.losses.mean_squared_error(tf.ones_like(dis_output_fake), dis_output_fake, scope='loss_gen')
        loss_pixel = tf.losses.absolute_difference(self.input_gray, gen_output, scope='loss_pixel')  # gen的输出为gray, 所以做L1 loss的时候也要用input_gray
        self.loss_G = loss_gen + lambda_pixel * loss_pixel
        # self.loss_G = loss_gen
        tf.summary.scalar('loss_gen', loss_gen)
        tf.summary.scalar('loss_pixel', loss_pixel)
        tf.summary.scalar('loss_G', self.loss_G)

        self.sample = self.gen_factor.create(self.input_ab, reuse_variable=True)   # input和output的channel改了，这边也得改下

        # optimizer
        global_step = tf.Variable(0, trainable=False)
        lr_decay = tf.train.exponential_decay(learning_rate=learning_rate,
                                              global_step=global_step,
                                              decay_steps=100000,
                                              decay_rate=0.96)

        self.optimizer_G = tf.train.AdamOptimizer(learning_rate=lr_decay).minimize(self.loss_G,
                                                                                   global_step=global_step,
                                                                                   var_list=self.gen_factor.var_list)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate=lr_decay).minimize(self.loss_D,
                                                                                   var_list=self.dis_factor.var_list)

        self.saver = tf.train.Saver()

    def train(self, epochs):
        print('training start')
        for epoch in range(epochs):
            # batch_list = self.train_data.get_batch_list()
            # batch_list_len = len(batch_list)
            batch_num = self.train_data.batch_num
            # for index, batch in enumerate(batch_list):
            for index in range(batch_num):
                # image_batch = self.train_data.get_image(batch)  # l: image_batch['l']  ab: image_batch['ab']
                image_batch = self.train_data.get_batch()
                feed_dic = {self.input_gray: image_batch['l'], self.input_ab: image_batch['ab']}
                _, loss_D = self.sess.run([self.optimizer_D, self.loss_D], feed_dict=feed_dic)
                _, loss_G = self.sess.run([self.optimizer_G, self.loss_G], feed_dict=feed_dic)

                sys.stdout.write("\r[record_queue:%d, batch_queue:%d][Epoch %d/%d] [Batch %d/%d] [Loss D %f] [Loss G %f]" %
                                 (self.train_data.record_queue.qsize(), self.train_data.batch_queue.qsize(),
                                  epoch, epochs,
                                  index, self.train_data.batch_num,
                                  loss_D,
                                  loss_G))
                if index % 100 == 0:
                    summary_str = self.sess.run(self.merged, feed_dict=feed_dic)
                    self.writer.add_summary(summary_str, index + epoch * batch_num)
                if index % 500 == 0:
                    self.sampler(epoch, index)
            if epoch % 10 == 0:
                self.saver.save(self.sess, './save_model/save_model_ab2gray_pix2pixloss/model')

    def sampler(self, epoch, index):
        # batch_list = self.test_data.get_batch_list()
        # image_batch = self.test_data.get_image(batch_list[0])
        image_batch = self.test_data.get_batch()
        feed_dic = {self.input_gray: image_batch['l'], self.input_ab: image_batch['ab']}
        gen_output = self.sess.run(self.sample, feed_dict=feed_dic)
        img_fake_lab = np.concatenate(((gen_output + 1.0) * 50.0, image_batch['ab'] * 110), axis=-1)
        img_real_lab = np.concatenate(((image_batch['l'] + 1.0) * 50.0, image_batch['ab'] * 110), axis=-1)
        img_fake_rgb = []
        img_real_rgb = []
        for i in range(self.test_data.batch_size):
            img_fake_rgb.append(color.lab2rgb(img_fake_lab[i, :, :, :]))
            img_real_rgb.append(color.lab2rgb(img_real_lab[i, :, :, :]))
        img_fake_rgb = np.array(img_fake_rgb)
        img_real_rgb = np.array(img_real_rgb)
        # img_rgb = color.lab2rgb(img_lab)
        sample_batch = np.concatenate((img_real_rgb, img_fake_rgb), axis=0)
        io.imsave('./images/images_ab2gray_pix2pixloss/%s_%s.png' % (epoch, index), self.to_rgb(sample_batch, self.batch_size))
        # img_big = self.to_rgb(sample_batch, ncol=5)

    def to_rgb(self, image_batch, ncol):
        batch_size = image_batch.shape[0]
        image_real = image_batch[0:batch_size // 2, :, :, :]
        image_fake = image_batch[batch_size // 2:, :, :, :]
        image_size = image_batch.shape[2]
        nrow = (batch_size // ncol) // 2
        image_big = np.zeros(shape=(image_size * nrow * 2, image_size * ncol, 3))
        for num in range(batch_size // 2):
            img_real = image_real[num, :, :, :]
            img_fake = image_fake[num, :, :, :]
            image = np.concatenate((img_real, img_fake), axis=0)
            image_big[((num // ncol) * image_size * 2):((num // ncol) * image_size * 2 + image_size * 2),
                      ((num % ncol) * image_size):((num % ncol) * image_size + image_size), :] = image

        return image_big

