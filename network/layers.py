# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from network.class_func import SpectralNormalization
from utils.params import *

from tensorflow.keras.layers import Layer, Conv2D, Dense, Flatten
from tensorflow import pad

class Conv(Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0,
                 pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn :
            self.conv = SpectralNormalization(
                Conv2D(
                    filters=self.channels, kernel_size=self.kernel,
                    kernel_initializer=WEIGHT_INITIALIZER,
                    kernel_regularizer=WEIGHT_REGULARIZER,
                    strides=self.stride, use_bias=self.use_bias
                ),
                name='sn_' + self.name
            )
        else :
            self.conv = Conv2D(
                filters=self.channels, kernel_size=self.kernel,
                kernel_initializer=WEIGHT_INITIALIZER,
                kernel_regularizer=WEIGHT_REGULARIZER,
                strides=self.stride, use_bias=self.use_bias,
                name=self.name
            )

    def call(self, x, training=None, mask=None):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad_ = self.pad * 2
            else:
                pad_ = max(self.kernel - (h % self.stride), 0)

            pad_top = pad_ // 2
            pad_bottom = pad_ - pad_top
            pad_left = pad_ // 2
            pad_right = pad_ - pad_left

            if self.pad_type == 'reflect':
                x = pad(x, [[0, 0], [pad_top, pad_bottom],
                            [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                x = pad(x, [[0, 0], [pad_top, pad_bottom],
                            [pad_left, pad_right], [0, 0]])

        x = self.conv(x)

        return x

class FullyConnected(Layer):
    def __init__(self, units, use_bias=True, sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(
                Dense(self.units, kernel_initializer=WEIGHT_INITIALIZER,
                      kernel_regularizer=WEIGHT_REGULARIZER_FULLY,
                      use_bias=self.use_bias),
                name='sn_' + self.name
            )
        else:
            self.fc = Dense(self.units, kernel_initializer=WEIGHT_INITIALIZER,
                            kernel_regularizer=WEIGHT_REGULARIZER_FULLY,
                            use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = Flatten(x)
        x = self.fc(x)

        return x