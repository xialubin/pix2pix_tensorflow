import tensorflow as tf


class Generator_Unet(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, in_channel=2, out_channel=1, training=True):  # 因为输入是ab，输出是gray，所以in_channel和out_channel也改了一下
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.training = training
        self.var_list = []

    def create(self, inputs, reuse_variable=False):
        output = inputs
        with tf.variable_scope(self.name, reuse=reuse_variable):
            layers = []
            for index, kernel in enumerate(self.encoder_kernels):
                name = 'conv2d' + str(index)
                initializer = tf.variance_scaling_initializer()
                res = tf.layers.conv2d(output,
                                       filters=kernel[0],
                                       kernel_size=kernel[1],
                                       strides=(2, 2),
                                       padding='same',
                                       kernel_initializer=initializer,
                                       use_bias=False,
                                       name=name)
                if kernel[3] == 1:
                    res = tf.contrib.layers.instance_norm(res)
                res = tf.nn.leaky_relu(res, alpha=0.2)
                output = tf.nn.dropout(res, keep_prob=kernel[2], name='dropout_' + name)
                layers.append(output)

            for index, kernel in enumerate(self.decoder_kernels):
                name = 'deconv2d' + str(index)
                initializer = tf.variance_scaling_initializer()
                res = tf.layers.conv2d_transpose(output,
                                                 filters=kernel[0],
                                                 kernel_size=kernel[1],
                                                 strides=(2, 2),
                                                 padding='same',
                                                 kernel_initializer=initializer,
                                                 use_bias=False,
                                                 name=name)
                if kernel[3] == 1:
                    res = tf.contrib.layers.instance_norm(res)
                res = tf.nn.relu(res)
                res = tf.nn.dropout(res, keep_prob=kernel[2], name='dropout_' + name)
                output = tf.concat([res, layers[len(layers) - 2 - index]], axis=-1)

            upsample = tf.keras.layers.UpSampling2D()
            output = upsample.apply(output)
            initializer = tf.variance_scaling_initializer()
            res = tf.layers.conv2d(output,
                                   filters=self.out_channel,
                                   kernel_size=4,
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer=initializer,
                                   use_bias=False,
                                   name='last_conv')
            output = tf.nn.tanh(res)
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output


class Discriminator(object):
    def __init__(self, name, kernels, in_channel=3, out_channel=1, training=True):
        self.name = name
        self.kernels = kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.training = training
        self.var_list = []

    def create(self, inputs, reuse_variable=False):
        output = inputs
        with tf.variable_scope(self.name, reuse=reuse_variable):
            for index, kernel in enumerate(self.kernels):
                initializer = tf.variance_scaling_initializer()
                res = tf.layers.conv2d(output,
                                       filters=kernel[0],
                                       kernel_size=kernel[1],
                                       strides=(2, 2),
                                       padding='same',
                                       kernel_initializer=initializer,
                                       name='conv' + str(index))
                res = tf.contrib.layers.instance_norm(res)
                output = tf.nn.leaky_relu(res, alpha=0.2)

            initializer = tf.variance_scaling_initializer()
            output = tf.layers.conv2d(output,
                                      filters=self.out_channel,
                                      kernel_size=4,
                                      kernel_initializer=initializer,
                                      padding='same',
                                      use_bias=False,
                                      name='last_conv')

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return output
