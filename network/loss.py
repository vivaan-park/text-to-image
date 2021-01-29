# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from network.utils import Relu, func_attention, cosine_similarity

from tensorflow import (nn, function, reduce_mean, square, math, transpose,
                        zeros_like, ones_like, expand_dims, matmul, tile,
                        reshape, exp, reduce_sum, cast, int32, concat,
                        float32, where, equal, constant)
import numpy as np

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

##############################################################################
# NLP Loss Function
##############################################################################

def caption_loss(cap_output, captions):
    loss = nn.softmax_cross_entropy_with_logits(logits=cap_output,
                                                labels=captions)
    loss = reduce_mean(loss)

    return loss

def word_level_correlation_loss(img_feature, word_emb,
                                gamma1=4.0, gamma2=5.0):
    batch_size = img_feature.shape[0]
    seq_len = word_emb.shape[1]
    similar_list = []

    for i in range(batch_size) :
        context = expand_dims(img_feature[i], axis=0)
        word = expand_dims(word_emb[i], axis=0)

        weighted_context, attn = func_attention(context, word, gamma1)

        aver_word = reduce_mean(word, axis=1, keepdims=True)

        res_word = matmul(aver_word, word, transpose_b=True)
        res_word_softmax = nn.softmax(res_word, axis=1)
        res_word_softmax = tile(res_word_softmax,
                                multiples=[1, weighted_context.shape[1], 1])

        self_weighted_context = transpose(weighted_context * res_word_softmax,
                                          perm=[0, 2, 1])

        word = reshape(word, [seq_len, -1])
        self_weighted_context = reshape(self_weighted_context, [seq_len, -1])

        row_sim = cosine_similarity(word, self_weighted_context)

        row_sim = exp(row_sim * gamma2)
        row_sim = reduce_sum(row_sim)
        row_sim = math.log(row_sim)

        similar_list.append(row_sim)

    word_match_loss = reduce_mean(
        nn.sigmoid_cross_entropy_with_logits(logits=similar_list,
                                             labels=ones_like(similar_list))
    )
    word_mismatch_loss = reduce_mean(
        nn.sigmoid_cross_entropy_with_logits(logits=similar_list,
                                             labels=zeros_like(similar_list))
    )

    loss = (word_match_loss + word_mismatch_loss) / 2.0

    return loss

def word_loss(img_feature, word_emb, class_id, gamma2=5.0):
    batch_size = word_emb.shape[0]
    seq_len = word_emb.shape[1]

    label = cast(range(batch_size), int32)
    masks = []
    similarities = []

    for i in range(batch_size):
        mask = (class_id.numpy() == class_id[i].numpy()).astype(np.uint8)
        mask[i] = 0
        masks.append(np.reshape(mask, newshape=[1, -1]))

        word = expand_dims(word_emb[i, :, :], axis=0)
        word = tile(word, multiples=[batch_size, 1, 1])

        context = img_feature

        weiContext, _ = func_attention(context, word)
        weiContext = transpose(weiContext, perm=[0, 2, 1])

        word = reshape(word, shape=[batch_size * seq_len, -1])
        weiContext = reshape(weiContext, shape=[batch_size * seq_len, -1])

        row_sim = cosine_similarity(word, weiContext)
        row_sim = reshape(row_sim, shape=[batch_size, seq_len])

        row_sim = exp(row_sim * gamma2)
        row_sim = reduce_sum(row_sim, axis=-1, keepdims=True)
        row_sim = math.log(row_sim)

        similarities.append(row_sim)

    similarities = concat(similarities, axis=-1)
    masks = cast(concat(masks, axis=0), float32)

    similarities = similarities * gamma2

    similarities = where(
        equal(masks, True),
        x=constant(-float('inf'), dtype=float32, shape=masks.shape),
        y=similarities
    )

    similarities1 = transpose(similarities, perm=[1, 0])

    loss0 = reduce_mean(
        nn.sparse_softmax_cross_entropy_with_logits(logits=similarities,
                                                    labels=label)
    )
    loss1 = reduce_mean(
        nn.sparse_softmax_cross_entropy_with_logits(logits=similarities1,
                                                    labels=label)
    )

    loss = loss0 + loss1

    return loss