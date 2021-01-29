# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.layers import Wrapper, Layer, BatchNormalization
from tensorflow import (initializers, float32, VariableAggregation,
                        reshape, matmul, transpose, reduce_sum)

##############################################################################
# Class function
##############################################################################

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

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name=self.name + '_u',
            dtype=float32,
            aggregation=VariableAggregation.ONLY_FIRST_REPLICA
        )

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        return output

    def update_weights(self):
        w_reshaped = reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = None

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = matmul(u_hat, transpose(w_reshaped))
                v_hat = v_ / (reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = matmul(v_hat, w_reshaped)
                u_hat = u_ / (reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = matmul(matmul(v_hat, w_reshaped), transpose(u_hat))
        self.u.assign(u_hat)

        self.layer.kernel = self.w / sigma

    def restore_weights(self):
        self.layer.kernel = self.w

##############################################################################
# Normalization
##############################################################################

class BatchNorm(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name='BatchNorm'):
        super(BatchNorm, self).__init__(name=name)
        self.momentum = momentum
        self.epsilon = epsilon

    def call(self, x, training=None, mask=None):
        x = BatchNormalization(
            momentum=self.momentum, epsilon=self.epsilon,
            center=True, scale=True, name=self.name
        )(x, training=training)
        return x

##############################################################################
# Activation Function
##############################################################################

class GLU(Layer):
    def __init__(self):
        super(GLU, self).__init__()