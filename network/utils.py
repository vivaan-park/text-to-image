# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.layers import (Wrapper, Layer, BatchNormalization,
                                     LeakyReLU, Activation, Dropout)
from tensorflow import (initializers, float32, VariableAggregation,
                        reshape, matmul, transpose, reduce_sum, sigmoid,
                        image, equal, argmax, reduce_mean, cast, nn)
from tensorflow.keras.activations import relu, tanh

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

    def build(self, input_shape):
        assert input_shape[-1] % 2 == 0, 'channels dont divide 2!'
        self.n_dim = len(input_shape)
        self.output_dim = input_shape[-1] // 2

    def call(self, x, training=None, mask=None):
        nc = self.output_dim
        if self.n_dim == 4:
            return x[:, :, :, :nc] * sigmoid(x[:, :, :, nc:])
        if self.n_dim == 3:
            return x[:, :, :nc] * sigmoid(x[:, :, nc:])
        if self.n_dim == 2:
            return x[:, :nc] * sigmoid(x[:, nc:])

def Leaky_Relu(x=None, alpha=0.01, name='leaky_relu'):
    if x is None:
        return LeakyReLU(alpha=alpha, name=name)
    else:
        return LeakyReLU(alpha=alpha, name=name)(x)

def Relu(x=None, name='relu'):
    if x is None:
        return Activation(relu, name=name)
    else:
        return Activation(relu, name=name)(x)

def Tanh(x=None, name='tanh'):
    if x is None:
        return Activation(tanh, name=name)
    else:
        return Activation(tanh, name=name)(x)

##############################################################################
# Pooling
##############################################################################

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return image.resize(x, size=new_size,
                        method=image.ResizeMethod.NEAREST_NEIGHBOR)

##############################################################################
# DropOut
##############################################################################

class DropOut(Layer):
    def __init__(self, drop_rate=0.5, name='DropOut'):
        super(DropOut, self).__init__(name=name)
        self.drop_rate = drop_rate

    def call(self, x, training=None, mask=None):
        x = Dropout(self.drop_rate, name=self.name)(x, training=training)
        return x

##############################################################################
# Accuracy
##############################################################################

def get_accuracy(logit, label):
    prediction = equal(argmax(logit, -1), argmax(label, -1))
    accuracy = reduce_mean(cast(prediction, float32))

    return accuracy

##############################################################################
# Natural Language Processing
##############################################################################

def func_attention(img_feature, word_emb, gamma1=4.0):
    bs, seq_len = word_emb.shape[0], word_emb.shape[1]
    h, w = img_feature.shape[1], img_feature.shape[2]
    hw = h * w

    context = reshape(img_feature, [bs, hw, -1])
    attn = matmul(context, word_emb, transpose_b=True)
    attn = reshape(attn, [bs*hw, seq_len])
    attn = nn.softmax(attn)

    attn = reshape(attn, [bs, hw, seq_len])
    attn = transpose(attn, perm=[0, 2, 1])
    attn = reshape(attn, [bs*seq_len, hw])

    attn = attn * gamma1
    attn = nn.softmax(attn)
    attn = reshape(attn, [bs, seq_len, hw])

    weighted_context = matmul(context, attn, transpose_a=True,
                              transpose_b=True)

    return weighted_context, reshape(transpose(attn, [0, 2, 1]),
                                     [bs, h, w, seq_len])