# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow import nn

def regularization_loss(model):
    loss = nn.scale_regularization_loss(model.losses)

    return loss