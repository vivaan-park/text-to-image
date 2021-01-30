# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Reshape
from tensorflow.keras.applications.inception_v3 import (preprocess_input,
                                                        InceptionV3)
from tensorflow import (image, equal, reshape, expand_dims, squeeze,
                        matmul, tile, where, constant, float32, nn,
                        transpose, concat)

from network.layers import Conv, FullyConnected
from network.nlp import VariousRNN, EmbedSequence
from network.utils import (DropOut, Relu, BatchNorm, GLU, nearest_up_sample,
                           Tanh)
from network.loss import reparametrize
from network.blocks import ResBlock

class CnnEncoder(Model):
    def __init__(self, embed_dim, name='CnnEncoder'):
        super(CnnEncoder, self).__init__(name=name)
        self.embed_dim = embed_dim

        self.inception_v3_preprocess = preprocess_input
        self.inception_v3 = InceptionV3(weights='imagenet',
                                        include_top=False, pooling='avg')
        self.inception_v3.trainable = False

        self.inception_v3_mixed7 = Model(
            inputs=self.inception_v3.input,
            outputs=self.inception_v3.get_layer('mixed7').output
        )
        self.inception_v3_mixed7.trainable = False

        self.emb_feature = Conv(channels=self.embed_dim, kernel=1, stride=1,
                                use_bias=False, name='emb_feature_conv')
        self.emb_code = FullyConnected(units=self.embed_dim,
                                       use_bias=True, name='emb_code_fc')

    def call(self, x, training=True, mask=None):
        x = ((x + 1) / 2) * 255.0
        x = image.resize(x, size=[299, 299],
                         method=image.ResizeMethod.BILINEAR)
        x = self.inception_v3_preprocess(x)

        code = self.inception_v3(x)
        feature = self.inception_v3_mixed7(x)

        feature = self.emb_feature(feature)
        code = self.emb_code(code)

        return feature, code

class RnnEncoder(Model):
    def __init__(self, n_words, embed_dim=256, drop_rate=0.5,
                 n_hidden=128, n_layer=1, bidirectional=True,
                 rnn_type='lstm', name='RnnEncoder'):
        super(RnnEncoder, self).__init__(name=name)
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        self.model = self.architecture()
        self.rnn = VariousRNN(self.n_hidden, self.n_layer, self.drop_rate,
                              self.bidirectional, rnn_type=self.rnn_type,
                              name=self.rnn_type + '_rnn')

    def architecture(self):
        model = []
        model += [EmbedSequence(self.n_words, self.embed_dim,
                                name='embed_layer')]
        model += [DropOut(self.drop_rate, name='dropout')]
        model = Sequential(model)

        return model

    def call(self, caption, training=True, mask=None):
        x = self.model(caption, training=training)
        word_emb, sent_emb = self.rnn(x, training=training)
        mask = equal(caption, 0)

        return word_emb, sent_emb, mask

class CA_NET(Model):
    def __init__(self, c_dim, name='CA_NET'):
        super(CA_NET, self).__init__(name=name)
        self.c_dim = c_dim

        self.model = self.architecture()

    def architecture(self):
        model = []
        model += [FullyConnected(units=self.c_dim * 2, name='mu_fc')]
        model += [Relu()]
        model = Sequential(model)

        return model

    def call(self, sent_emb, training=True, mask=None):
        x = self.model(sent_emb, training=training)

        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]

        c_code = reparametrize(mu, logvar)

        return c_code, mu, logvar

class SpatialAttention(Layer):
    def __init__(self, channels, name='SpatialAttention'):
        super(SpatialAttention, self).__init__(name=name)
        self.channels = channels

        self.word_conv = Conv(self.channels, kernel=1, stride=1,
                              use_bias=False, name='word_conv')
        self.sentence_fc = FullyConnected(units=self.channels, name='sent_fc')
        self.sentence_conv = Conv(self.channels, kernel=1, stride=1,
                                  use_bias=False, name='sentence_conv')

    def build(self, input_shape):
        self.bs, self.h, self.w, _ = input_shape[0]
        self.hw = self.h * self.w
        self.seq_len = input_shape[2][1]

    def call(self, inputs, training=True):
        x, sentence, context, mask = inputs
        x = reshape(x, shape=[self.bs, self.hw, -1])

        context = expand_dims(context, axis=1)
        context = self.word_conv(context)
        context = squeeze(context, axis=1)

        attn = matmul(x, context, transpose_b=True)
        attn = reshape(attn, shape=[self.bs * self.hw, self.seq_len])

        mask = tile(mask, multiples=[self.hw, 1])
        attn = where(
            equal(mask, True),
            x=constant(-float('inf'), dtype=float32, shape=mask.shape),
            y=attn
        )
        attn = nn.softmax(attn)
        attn = reshape(attn, shape=[self.bs, self.hw, self.seq_len])

        weighted_context = matmul(context, attn, transpose_a=True,
                                  transpose_b=True)
        weighted_context = reshape(transpose(weighted_context, perm=[0, 2, 1]),
                                   shape=[self.bs, self.h, self.w, -1])
        word_attn = reshape(attn, shape=[self.bs, self.h, self.w, -1])

        return weighted_context, word_attn

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

class Generator_64(Layer):
    def __init__(self, channels, name='Generator_64'):
        super(Generator_64, self).__init__(name=name)
        self.channels = channels

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []
        model += [FullyConnected(units=self.channels * 4 * 4 * 2,
                                 use_bias=False, name='code_fc')]
        model += [BatchNorm(name='batch_norm')]
        model += [GLU()]
        model += [Reshape(target_shape=[4, 4, self.channels])]

        for i in range(4):
            model += [UpBlock(self.channels // 2, name='up_block_' + str(i))]
            self.channels = self.channels // 2

        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1,
                                    pad_type='reflect', use_bias=False,
                                    name='g_64_logit')]
        generate_img_block += [Tanh()]
        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block

    def call(self, c_z_code, training=True, mask=None):
        h_code = self.model(c_z_code, training=training)
        x = self.generate_img_block(h_code, training=training)

        return h_code, x

class Generator_128(Layer):
    def __init__(self, channels, name='Generator_128'):
        super(Generator_128, self).__init__(name=name)
        self.channels = channels

        self.spatial_attention = SpatialAttention(channels=self.channels)

        self.model, self.generate_img_block = self.architecture()

    def architecture(self):
        model = []
        for i in range(2):
            model += [ResBlock(self.channels * 2, name='resblock_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]
        model = Sequential(model)

        generate_img_block = []
        generate_img_block += [Conv(channels=3, kernel=3, stride=1, pad=1,
                                    pad_type='reflect', use_bias=False,
                                    name='g_128_logit')]
        generate_img_block += [Tanh()]
        generate_img_block = Sequential(generate_img_block)

        return model, generate_img_block

    def call(self, inputs, training=True):
        h_code, c_code, word_emb, mask = inputs
        c_code, _ = self.spatial_attention([h_code, c_code, word_emb, mask])

        h_c_code = concat([h_code, c_code], axis=-1)

        h_code = self.model(h_c_code, training=training)
        x = self.generate_img_block(h_code)

        return c_code, h_code, x

class Generator_256(Layer):
    def __init__(self, channels, name='Generator_256'):
        super(Generator_256, self).__init__(name=name)
        self.channels = channels

        self.spatial_attention = SpatialAttention(channels=self.channels)
        self.model = self.architecture()

    def architecture(self):
        model = []
        for i in range(2):
            model += [ResBlock(self.channels * 2, name='res_block_' + str(i))]

        model += [UpBlock(self.channels, name='up_block')]
        model += [Conv(channels=3, kernel=3, stride=1, pad=1,
                       pad_type='reflect', use_bias=False,
                       name='g_256_logit')]
        model += [Tanh()]
        model = Sequential(model)

        return model