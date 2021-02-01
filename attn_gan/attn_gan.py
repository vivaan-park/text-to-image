# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from tensorflow import (data, GradientTape, random, int32, gather, image,
                        reduce_sum)
from tensorflow.python.data.experimental import (prefetch_to_device,
                                                 shuffle_and_repeat,
                                                 map_and_batch)
from tensorflow.keras.optimizers import Adam
from tensorflow import train as tf_train

from utils.others import check_folder
from data.preprocess import Image_data
from attn_gan.generator import (RnnEncoder, CnnEncoder, CA_NET, Generator)
from attn_gan.discriminator import Discriminator
from network.loss import (word_loss, sent_loss, discriminator_loss,
                          generator_loss, kl_loss)

class AttnGAN():
    def __init__(self, args):

        self.phase = args.phase
        self.model_name = 'AttnGAN'

        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr

        self.gan_type = args.gan_type

        self.d_dim = args.d_dim
        self.g_dim = args.g_dim
        self.embed_dim = args.embed_dim
        self.z_dim = args.z_dim

        self.adv_weight = args.adv_weight
        self.kl_weight = args.kl_weight
        self.embed_weight = args.embed_weight

        self.sn = args.sn

        self.img_height = args.img_height
        self.img_width = args.img_width

        self.img_ch = args.img_ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.dataset_path = os.path.join('./dataset', self.dataset_name)

    def build_model(self):
        img_data_class = Image_data(self.img_height, self.img_width,
                                    self.img_ch, self.dataset_path,
                                    self.augment_flag)
        train_class_id, train_captions, train_images,\
        test_captions, test_images, idx_to_word, word_to_idx = \
            img_data_class.preprocess()
        self.vocab_size = len(idx_to_word)
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

        if self.phase == 'train' :
            self.dataset_num = len(train_images)

            img_and_caption = data.Dataset.from_tensor_slices(
                (train_images, train_captions, train_class_id)
            )

            gpu_device = '/gpu:0'
            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)
            ).apply(
                map_and_batch(
                    img_data_class.image_processing,
                    batch_size=self.batch_size,
                    num_parallel_batches=16,
                    drop_remainder=True)
            ).apply(prefetch_to_device(gpu_device, None))

            self.img_caption_iter = iter(img_and_caption)

            self.rnn_encoder = RnnEncoder(
                n_words=self.vocab_size, embed_dim=self.embed_dim,
                drop_rate=0.5, n_hidden=128, n_layer=1,
                bidirectional=True, rnn_type='lstm'
            )
            self.cnn_encoder = CnnEncoder(embed_dim=self.embed_dim)

            self.ca_net = CA_NET(c_dim=self.z_dim)
            self.generator = Generator(channels=self.g_dim)

            self.discriminator = Discriminator(
                channels=self.d_dim, embed_dim=self.embed_dim
            )

            """ Optimizer """
            self.g_optimizer = Adam(learning_rate=self.init_lr, beta_1=0.5,
                                    beta_2=0.999, epsilon=1e-08)

            d_64_optimizer = Adam(learning_rate=self.init_lr, beta_1=0.5,
                                  beta_2=0.999, epsilon=1e-08)
            d_128_optimizer = Adam(learning_rate=self.init_lr, beta_1=0.5,
                                   beta_2=0.999, epsilon=1e-08)
            d_256_optimizer = Adam(learning_rate=self.init_lr, beta_1=0.5,
                                   beta_2=0.999, epsilon=1e-08)
            self.d_optimizer = [d_64_optimizer, d_128_optimizer,
                                d_256_optimizer]

            self.embed_optimizer = Adam(learning_rate=self.init_lr,
                                        beta_1=0.5, beta_2=0.999,
                                        epsilon=1e-08)

            self.ckpt = tf_train.Checkpoint(
                rnn_encoder=self.rnn_encoder,
                cnn_encoder=self.cnn_encoder,
                ca_net=self.ca_net,
                generator=self.generator,
                discriminator=self.discriminator,
                g_optimizer=self.g_optimizer,
                d_64_optimizer=d_64_optimizer,
                d_128_optimizer=d_128_optimizer,
                d_256_optimizer=d_256_optimizer,
                embed_optimizer=self.embed_optimizer
            )
            self.manager = tf_train.CheckpointManager(
                self.ckpt, self.checkpoint_dir, max_to_keep=2
            )
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint)
                self.start_iteration = int(
                    self.manager.latest_checkpoint.split('-')[-1]
                )
                print('Latest checkpoint restored!!')
                print('start iteration :', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')
        else :
            self.dataset_num = len(test_captions)

            gpu_device = '/gpu:0'
            img_and_caption = data.Dataset.from_tensor_slices(
                (test_images, test_captions)
            )

            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing,
                              batch_size=self.batch_size,
                              num_parallel_batches=16,
                              drop_remainder=True)).apply(
                prefetch_to_device(gpu_device, None))

            self.img_caption_iter = iter(img_and_caption)

            self.rnn_encoder = RnnEncoder(
                n_words=self.vocab_size, embed_dim=self.embed_dim,
                drop_rate=0.5, n_hidden=128, n_layer=1,
                bidirectional=True, rnn_type='lstm'
            )
            self.ca_net = CA_NET(c_dim=self.z_dim)
            self.generator = Generator(channels=self.g_dim)

            self.ckpt = tf_train.Checkpoint(rnn_encoder=self.rnn_encoder,
                                            ca_net = self.ca_net,
                                            generator=self.generator)
            self.manager = tf_train.CheckpointManager(
                self.ckpt, self.checkpoint_dir, max_to_keep=2
            )
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(
                    self.manager.latest_checkpoint
                ).expect_partial()
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

    def embed_train_step(self, real_256, caption, class_id):
        with GradientTape() as embed_tape:
            target_sentence_index = random.uniform(
                shape=[], minval=0, maxval=10, dtype=int32
            )
            caption = gather(caption, target_sentence_index, axis=1)

            word_feature, sent_code = self.cnn_encoder(
                real_256, training=True
            )
            word_emb, sent_emb, mask = self.rnn_encoder(
                caption, training=True
            )

            w_loss = word_loss(word_feature, word_emb, class_id)
            s_loss = sent_loss(sent_code, sent_emb, class_id)

            embed_loss = self.embed_weight * (w_loss + s_loss)

        embed_train_variable = self.cnn_encoder.trainable_variables + \
                               self.rnn_encoder.trainable_variables
        embed_gradient = embed_tape.gradient(
            embed_loss, embed_train_variable
        )
        self.embed_optimizer.apply_gradients(
            zip(embed_gradient, embed_train_variable)
        )

        return embed_loss

    def d_train_step(self, real_256, caption):
        with GradientTape() as d_64_tape, GradientTape() as d_128_tape,\
                GradientTape() as d_256_tape :
            target_sentence_index = random.uniform(shape=[], minval=0,
                                                   maxval=10, dtype=int32)
            caption = gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(
                caption, training=True
            )
            z_code = random.normal(shape=[self.batch_size, self.z_dim])
            c_code, mu, logvar = self.ca_net(sent_emb, training=True)
            fake_imgs = self.generator([c_code, z_code, word_emb, mask],
                                       training=True)

            real_64 = image.resize(real_256, target_size=[64, 64],
                                   method=image.ResizeMethod.BILINEAR)
            real_128 = image.resize(real_256, target_size=[128, 128],
                                    method=image.ResizeMethod.BILINEAR)
            fake_64, fake_128, fake_256 = fake_imgs

            uncond_real_logits, cond_real_logits = \
                self.discriminator([real_64, real_128, real_256, sent_emb],
                                   training=True)
            _, cond_wrong_logits = self.discriminator(
                [real_64[:(self.batch_size - 1)],
                 real_128[:(self.batch_size - 1)],
                 real_256[:(self.batch_size - 1)],
                 sent_emb[1:self.batch_size]]
            )
            uncond_fake_logits, cond_fake_logits = self.discriminator(
                [fake_64, fake_128, fake_256, sent_emb], training=True
            )

            d_adv_loss = []

            for i in range(3):
                uncond_real_loss, uncond_fake_loss = discriminator_loss(
                    self.gan_type,
                    uncond_real_logits[i],
                    uncond_fake_logits[i]
                )
                cond_real_loss, cond_fake_loss = discriminator_loss(
                    self.gan_type,
                    cond_real_logits[i],
                    cond_fake_logits[i]
                )
                _, cond_wrong_loss = discriminator_loss(
                    self.gan_type,
                    None,
                    cond_wrong_logits[i]
                )

                each_d_adv_loss = self.adv_weight * (
                        ((uncond_real_loss + cond_real_loss) / 2) +
                        (uncond_fake_loss + cond_fake_loss + cond_wrong_loss) / 3
                )
                d_adv_loss.append(each_d_adv_loss)

            d_loss = reduce_sum(d_adv_loss)

        d_train_variable = [self.discriminator.d_64.trainable_variables,
                            self.discriminator.d_128.trainable_variables,
                            self.discriminator.d_256.trainable_variables]
        d_tape = [d_64_tape, d_128_tape, d_256_tape]

        for i in range(3):
            d_gradient = d_tape[i].gradient(
                d_adv_loss[i], d_train_variable[i]
            )
            self.d_optimizer[i].apply_gradients(
                zip(d_gradient, d_train_variable[i])
            )

        return d_loss, reduce_sum(d_adv_loss)

    def g_train_step(self, caption, class_id):
        with GradientTape() as g_tape:
            target_sentence_index = random.uniform(shape=[], minval=0,
                                                   maxval=10, dtype=int32)
            caption = gather(caption, target_sentence_index, axis=1)

            word_emb, sent_emb, mask = self.rnn_encoder(
                caption, training=True
            )

            z_code = random.normal(shape=[self.batch_size, self.z_dim])
            c_code, mu, logvar = self.ca_net(sent_emb, training=True)

            fake_imgs = self.generator([c_code, z_code, word_emb, mask],
                                       training=True)
            fake_64, fake_128, fake_256 = fake_imgs

            uncond_fake_logits, cond_fake_logits = \
                self.discriminator([fake_64, fake_128, fake_256, sent_emb],
                                   training=True)

            g_adv_loss = 0

            for i in range(3):
                g_adv_loss += self.adv_weight * \
                              (generator_loss(self.gan_type, uncond_fake_logits[i]) +
                               generator_loss(self.gan_type, cond_fake_logits[i]))

            word_feature, sent_code = self.cnn_encoder(fake_256, training=True)

            w_loss = word_loss(word_feature, word_emb, class_id)
            s_loss = sent_loss(sent_code, sent_emb, class_id)

            g_embed_loss = self.embed_weight * (w_loss + s_loss) * 5.0

            g_kl_loss = self.kl_weight * kl_loss(mu, logvar)

            g_loss = g_adv_loss + g_kl_loss + g_embed_loss

        g_train_variable = self.generator.trainable_variables + \
                           self.ca_net.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        return g_loss, g_adv_loss, g_kl_loss, g_embed_loss, fake_256