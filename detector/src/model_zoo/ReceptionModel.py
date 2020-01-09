import tensorflow as tf

class Conv2D(tf.keras.layers.Layer):
    """Utility function to apply conv + BN.
        # Arguments
            filters: filters in `Conv2D`.
            kernel_size: kernel size as in `Conv2D`.
            strides: strides in `Conv2D`.
            padding: padding mode in `Conv2D`.
            activation: activation in `Conv2D`.
            use_bias: whether to use a bias in `Conv2D`.
            momentum: momentum in `BatchNormalization`
            epsilon: epsilon in `BatchNormalization`
            scale: whether to use scale in `BatchNormalization`
            rate: Float between 0 and 1. Fraction of the units to drop.
            name: name of the ops; will become `name + '_ac'` for the activation, `name + '_cn'` for the convolutional layer and `name + '_bn'` for the batch norm layer.
        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    # TODO Check defaults with the research [paper](https://arxiv.org/abs/1602.07261)
    def __init__(self, filters, kernel_size, strides = (1, 1), padding = 'same', activation = None, use_bias = True, momentum = 0.99, epsilon = 0.001, scale = False, rate = 0, name = None, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides=strides
        self.padding=padding
        self.activation=activation
        self.bias=use_bias
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.rate = rate
        self.name = name

    def build(self):
        # TODO wrap it in TimeDistributed layer and check the flow
        # self.conv = tf.keras.layers.TimeDistributed(tf.keras.layers.SeparableConv2D(self.filters, self.kernel_size, name = self.name + '_cn'), input_shape=(37, 100, 100, 3))
        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, name = self.name + '_cn')
        # TODO Try fused option for `BatchNorm`
        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.bn = tf.keras.layers.BatchNormalization(axis = channel_axis , scale = self.scale, name = self.name + '_bn')
        if self.rate > 0:
            self.drop = tf.keras.layers.SpatialDropout2D(self.rate)
        if self.activation is not None:
            ac_name = None if name is None else name + '_ac'
            self.activate = layers.Activation(activation, name=ac_name)

    def call(self, inputs, training = False):
        x = self.conv(inputs)
        # TODO Check whether `BatchNorm` is effective before or after `SpatialDropout`
        if training:
            x = self.drop(x, training = training)
        x = self.bn(x)
        if self.activate:
            x = self.activate(x)
        return x

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'strides': self.strides,
                       'padding': self.padding,
                       'activation': self.activation,
                       'use_bias': self.bias,
                       'momentum':self.momentum,
                       'epsilon': self.epsilon,
                       'scale': self.scale,
                       'rate': self.rate,
                       'name': self.name})
        return config

class ReceptionStem(tf.keras.Model):
    def __init__(self, filters = 32, input_shape = (37, 100, 100, 3)):
        super(ReceptionBlockA, self).__init__(name = "Reception_Stem")
        self.input = tf.keras.layers.InputLayer(input_shape=(37, 100, 100, 3))
        self.conv1a = Conv2D(filters, (3, 3), strides=2, padding='valid')
        self.conv1b = Conv2D(filters, (3, 3), padding='valid')

        self.conv2a = Conv2D(2 * filters, (3, 3), padding='same')
        self.conv2b = Conv2D(2 * filters, (3, 3), strides=2, padding='valid')

        self.conv3 = Conv2D((2 * filters) + (filters // 2), (1, 1))
        self.conv4 = Conv2D(6 * filters, (3, 3), padding='valid')
        self.conv5 = Conv2D(8 * filters, (3, 3), strides=2, padding='valid')

    def call(self, inputs):
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class ReceptionBlockA(tf.keras.Model):
    def __init__(self, filters = 32, name = ''):
        super(ReceptionBlockA, self).__init__(name = "Reception_BlockA" + name)
        self.activation = tf.keras.layers.Activation('relu')
        self.conv1a = Conv2D(filters, (1, 1))
        self.conv1b = Conv2D(filters, (1, 1))
        self.conv1c = Conv2D(filters, (1, 1))
        self.conv2a = Conv2D(filters, (3, 3))
        self.conv2b = Conv2D(filters + (filters // 2), (3, 3))
        self.conv2c = Conv2D(2 * filters, (1, 1))
        self.conv1d = Conv2D(12 * filters, (1, 1))

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.concat0 = tf.keras.layers.Concatenate(channel_axis)
        self.concat1 = tf.keras.layers.Concatenate(channel_axis)

    def call(self, inputs):
        x0 = self.conv1a(inputs)
        x1 = self.conv1b(inputs)
        x2 = self.conv1c(inputs)

        x1 = self.conv2a(x1)
        x2 = self.conv2b(x2)
        x2 = self.conv2c(x2)

        x = self.concat0([x0, x1, x2])
        x = self.conv1d(x)
        x = self.concat1([x, input])
        return self.activation(x)

class ReceptionBlockB(tf.keras.Model):
    def __init__(self, filters = 128, name = ''):
        super(ReceptionBlockB, self).__init__(name = "Reception_BlockB" + name)
        self.activation = tf.keras.layers.Activation('relu')

        self.conv1a = Conv2D(filters, (1, 1))
        self.conv1b = Conv2D(filters + (filters // 2), (1, 1))

        self.conv2a = Conv2D(filters + (filters // 4), (1, 7))
        self.conv2b = Conv2D(filters + (filters // 2), (7, 1))

        # TODO check what 1154 Linear units mean according to [paper](https://arxiv.org/abs/1602.07261) for 17*17 grid (Inception-ResNet-B-v2)
        self.conv3 = Conv2D(9 * filters, (1, 1))

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.concat0 = tf.keras.layers.Concatenate(channel_axis)
        self.concat1= tf.keras.layers.Concatenate(channel_axis)

    def call(self, inputs):
        x0 = self.conv1a(inputs)
        x1 = self.conv1b(inputs)

        x1 = self.conv2a(x1)
        x1 = self.conv2b(x1)

        x = self.concat0([x0, x1])
        x = self.conv3(x)
        x = self.concat1([x, input])
        return self.activation(x)

class ReceptionBlockC(tf.keras.Model):
    def __init__(self, filters = 192, name = ''):
        super(ReceptionBlockC, self).__init__(name = "Reception_BlockC" + name)
        self.activation = tf.keras.layers.Activation('relu')

        self.conv1a = Conv2D(filters, (1, 1))
        self.conv1b = Conv2D(filters, (1, 1))

        self.conv2a = Conv2D(filters + (filters // 6), (1, 3))
        self.conv2b = Conv2D(filters + (filters // 3), (3, 1))

        # TODO check what 2048 Linear units mean according to [paper](https://arxiv.org/abs/16$
        self.conv3 = Conv2D((10 * filters) + (2 * filter // 3), (1, 1))

        channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.concat0 = tf.keras.layers.Concatenate(channel_axis)
        self.concat1= tf.keras.layers.Concatenate(channel_axis)

    def call(self, inputs):
        x0 = self.conv1a(inputs)
        x1 = self.conv1b(inputs)

        x1 = self.conv2a(x1)
        x1 = self.conv2b(x1)

        x = self.concat0([x0, x1])
        x = self.conv3(x)
        x = self.concat1([x, input])
        return self.activation(x)

class ReceptionModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    #self.input = tf.keras.layers.InputLayer(input_shape=(37, 100, 100, 3))
    self.stem = ReceptionStem(input_shape=(37, 100, 100, 3))
    self.reca = ReceptionBlockA()
    self.reda = ReceptionReductionA()
    self.recb = ReceptionBlockB()
    self.redb = ReceptionReductionB()
    self.recc = ReceptionBlockC()
    self.avgpool = tf.keras.layers.GlobalAveragePooling3D()
    self.drop = tf.keras.layers.Dropout(rate=0.2, noise_shape=(, 37, 100, 100, 3)) # noise_shape=(batch_size, timesteps, features)
    self.flatten = tf.keras.layers.Flatten()
    self.output = tf.keras.layers.Dense(units=27, activation='softmax', name='prediction')

  def call(self, inputs, training=False):
    x = self.stem(inputs)
    x = self.reca(x)
    x = self.reda(x)
    x = self.recb(x)
    x = self.redb(x)
    x = self.recc(x)
    x = self.avgpool(x)
    if training:
        x = self.drop(x)
    x = self.flatten(x)
    return self.output(x)

