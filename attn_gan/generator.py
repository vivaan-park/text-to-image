# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow import image

from network.layers import Conv, FullyConnected
from network.nlp import VariousRNN

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