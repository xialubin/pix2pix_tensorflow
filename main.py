from train import Model
import tensorflow as tf

kernels_gen_encoder = [
    # out_channel, kernel_size, keep_drop, norm_or_not
    (64, 4, 1.0, 0),  # [batch, 256, 256, ch] => [batch, 128, 128, 64]
    (128, 4, 1.0, 1),  # [batch, 128, 128, 64] => [batch, 64, 64, 128]
    (256, 4, 1.0, 1),  # [batch, 64, 64, 128] => [batch, 32, 32, 256]
    (512, 4, 0.5, 1),  # [batch, 32, 32, 256] => [batch, 16, 16, 512]
    (512, 4, 0.5, 1),  # [batch, 16, 16, 512] => [batch, 8, 8, 512]
    (512, 4, 0.5, 1),  # [batch, 8, 8, 512] => [batch, 4, 4, 512]
    (512, 4, 0.5, 1),  # [batch, 4, 4, 512] => [batch, 2, 2, 512]
    (512, 4, 0.5, 0)  # [batch, 2, 2, 512] => [batch, 1, 1, 512]
]

kernels_gen_decoder = [
    # out_channel, kernel_size, keep_drop, norm_or_not
    (512, 4, 0.5, 1),  # [batch, 1, 1, 512] => [batch, 2, 2, 512+512]
    (512, 4, 0.5, 1),  # [batch, 2, 2, 1024] => [batch, 4, 4, 512+512]
    (512, 4, 0.5, 1),  # [batch, 4, 4, 1024] => [batch, 8, 8, 512+512]
    (512, 4, 0.5, 1),  # [batch, 8, 8, 1024] => [batch, 16, 16, 512+512]
    (256, 4, 1.0, 1),  # [batch, 16, 16, 1024] => [batch, 32, 32, 256+256]
    (128, 4, 1.0, 1),  # [batch, 32, 32, 512] => [batch, 64, 64, 128+128]
    (64, 4, 1.0, 1)  # [batch, 64, 64, 256] => [batch, 128, 128, 64+64]
]

kernels_dis = [
    # out_channel, kernel_size
    (64, 4),  # [batch, 256, 256, ch] => [batch, 128, 128, 64]
    (128, 4),  # [batch, 128, 128, 64] => [batch, 64, 64, 128]
    (256, 4),  # [batch, 64, 64, 128] => [batch, 32, 32, 256]
    (512, 4)  # [batch, 32, 32, 256] => [batch, 16, 16, 512]
]

with tf.device('gpu:1'):
    pix2pix = Model(kernels_gen_encoder, kernels_gen_decoder, kernels_dis, trainning=True)
    pix2pix.build(lambda_pixel=100.0)
    pix2pix.net_initial()
    pix2pix.train(epochs=201)
