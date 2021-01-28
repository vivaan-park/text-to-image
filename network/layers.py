# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import tensorflow as tf

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
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
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name=self.name + '_u',
            dtype=tf.float32, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

        super(SpectralNormalization, self).build()