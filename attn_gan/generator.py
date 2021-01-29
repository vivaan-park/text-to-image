# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3

from network.layers import Conv, FullyConnected

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