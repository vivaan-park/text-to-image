# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.layers import Wrapper, Layer

class SpectralNormalization(Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                f'`Layer` instance. You passed: {layer}')
        super(SpectralNormalization, self).__init__(layer, **kwargs)