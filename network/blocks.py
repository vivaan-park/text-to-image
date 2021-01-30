# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from network.layers import Conv
from network.utils import BatchNorm, GLU, nearest_up_sample

from tensorflow.keras.layers import Layer
from tensorflow import name_scope
from tensorflow.keras import Sequential

class ResBlock(Layer):
    def __init__(self, channels, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels = channels

        self.conv_0 = Conv(self.channels * 2, kernel=3, stride=1, pad=1,
                           pad_type='reflect', use_bias=False, name='conv_0')
        self.batch_norm_0 = BatchNorm(momentum=0.9, epsilon=1e-5,
                                      name='batch_norm_0')

        self.conv_1 = Conv(self.channels, kernel=3, stride=1, pad=1,
                           pad_type='reflect', use_bias=False,  name='conv_1')
        self.batch_norm_1 = BatchNorm(momentum=0.9, epsilon=1e-5,
                                      name='batch_norm_1')

    def call(self, x_init, training=None, mask=None):
        with name_scope(self.name):
            with name_scope('res1'):
                x = self.conv_0(x_init)
                x = self.batch_norm_0(x, training=training)
                x = GLU()(x)

            with name_scope('res2'):
                x = self.conv_1(x)
                x = self.batch_norm_1(x, training=training)

            return x + x_init

class UpBlock(Layer):
    def __init__(self, channels, name='UpBlock'):
        super(UpBlock, self).__init__(name=name)
        self.channels = channels

        self.model = self.architecture()

    def architecture(self):
        model = []
        model += [Conv(self.channels * 2, kernel=3, stride=1, pad=1,
                       pad_type='reflect', use_bias=False, name='conv')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]
        model = Sequential(model)

        return model

    def call(self, x_init, training=True):
        x = nearest_up_sample(x_init, scale_factor=2)
        x = self.model(x, training=training)

        return x