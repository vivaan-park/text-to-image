# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from network.layers import Conv
from network.utils import BatchNorm

from tensorflow.keras.layers import Layer

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