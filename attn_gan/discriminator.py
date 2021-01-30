# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential

from network.layers import Conv
from network.utils import Leaky_Relu, BatchNorm

class Discriminator_64(Layer):
    def __init__(self, channels, name='Discriminator_64'):
        super(Discriminator_64, self).__init__(name=name)
        self.channels = channels

        self.uncond_logit_conv = Conv(channels=1, kernel=4, stride=4,
                                      use_bias=True, name='uncond_d_logit')
        self.cond_logit_conv = Conv(channels=1, kernel=4, stride=4,
                                    use_bias=True, name='cond_d_logit')
        self.model, self.code_block = self.architecture()

    def architecture(self):
        model = []
        model += [Conv(self.channels, kernel=4, stride=2, pad=1,
                       pad_type='reflect', use_bias=False, name='convv')]
        model += [Leaky_Relu(alpha=0.2)]

        for i in range(3):
            model += [Conv(self.channels * 2, kernel=4, stride=2, pad=1,
                           pad_type='reflect', use_bias=False,
                           name='conv_' + str(i))]
            model += [BatchNorm(name=f'batch_norm_{str(i)}')]
            model += [Leaky_Relu(alpha=0.2)]

            self.channels = self.channels * 2

        model = Sequential(model)

        code_block = []
        code_block += [Conv(self.channels, kernel=3, stride=1, pad=1,
                            pad_type='reflect', use_bias=False,
                            name='conv_code')]
        code_block += [BatchNorm(name='batch_norm_code')]
        code_block += [Leaky_Relu(alpha=0.2)]
        code_block = Sequential(code_block)

        return model, code_block