# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from network.utils import Relu

from tensorflow import (nn, function, reduce_mean, square, math,
                        zeros_like, ones_like)

##############################################################################
# Loss Function
##############################################################################

def regularization_loss(model):
    loss = nn.scale_regularization_loss(model.losses)

    return loss

##############################################################################
# GAN Loss Function
##############################################################################

@function
def discriminator_loss(gan_type, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if real_logit is None :
        if gan_type == 'lsgan':
            fake_loss = reduce_mean(square(fake_logit))
        elif gan_type == 'gan':
            fake_loss = reduce_mean(
                nn.sigmoid_cross_entropy_with_logits(
                    labels=zeros_like(fake_logit), logits=fake_logit
                )
            )
        elif gan_type == 'hinge':
            fake_loss = reduce_mean(Relu(1 + fake_logit))
    else :
        if gan_type == 'lsgan':
            real_loss = reduce_mean(
                math.squared_difference(real_logit, 1.0)
            )
            fake_loss = reduce_mean(square(fake_logit))
        elif gan_type == 'gan':
            real_loss = reduce_mean(
                nn.sigmoid_cross_entropy_with_logits(labels=ones_like(real_logit),
                                                     logits=real_logit)
            )
            fake_loss = reduce_mean(
                nn.sigmoid_cross_entropy_with_logits(labels=zeros_like(fake_logit),
                                                     logits=fake_logit)
            )
        elif gan_type == 'hinge':
            real_loss = reduce_mean(Relu(1 - real_logit))
            fake_loss = reduce_mean(Relu(1 + fake_logit))

    return real_loss, fake_loss

@function
def generator_loss(gan_type, fake_logit):
    fake_loss = 0

    if gan_type == 'lsgan':
        fake_loss = reduce_mean(math.squared_difference(fake_logit, 1.0))
    elif gan_type == 'gan':
        fake_loss = reduce_mean(
            nn.sigmoid_cross_entropy_with_logits(labels=ones_like(fake_logit),
                                                 logits=fake_logit)
        )
    elif gan_type == 'hinge':
        fake_loss = -reduce_mean(fake_logit)

    return fake_loss

@function
def L2_loss(x, y):
    loss = reduce_mean(square(x - y))

    return loss

@function
def L1_loss(x, y):
    loss = reduce_mean(abs(x - y))

    return loss