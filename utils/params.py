# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow import initializers
from tensorflow.keras import regularizers

WEIGHT_INITIALIZER = initializers.RandomNormal(mean=0.0, stddev=0.02)
WEIGHT_REGULARIZER = regularizers.l2(0.0001)
WEIGHT_REGULARIZER_FULLY= regularizers.l2(0.0001)