# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import time
from collections import defaultdict

from tensorflow import (data, GradientTape, random, int32, gather, image,
                        reduce_sum, summary)
from tensorflow.python.data.experimental import shuffle_and_repeat, map_and_batch
from tensorflow.keras.optimizers import Adam
from tensorflow import train as tf_train
import numpy as np

from utils.others import check_folder
from data.preprocess import Image_data, return_images, save_images
from data.dataset import tokenizer
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
        self.valid_dir = args.valid_dir
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

        self.captions = args.captions

    def build_model(self):
        img_data_class = Image_data(self.img_height, self.img_width,
                                    self.img_ch, self.augment_flag)
        train_class_id, train_captions, train_images,\
        test_captions, test_images, idx_to_word, word_to_idx = \
            img_data_class.preprocess()
        self.vocab_size = len(idx_to_word)
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

        if self.phase == 'train':
            self.dataset_num = len(train_images)

            img_and_caption = data.Dataset.from_tensor_slices(
                (train_images, train_captions, train_class_id)
            )

            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing,
                              batch_size=self.batch_size,
                              num_parallel_batches=16,
                              drop_remainder=True))

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

        if self.phase == 'test':
            self.dataset_num = len(test_images)

            img_and_caption = data.Dataset.from_tensor_slices(
                (test_images, test_captions)
            )

            img_and_caption = img_and_caption.apply(
                shuffle_and_repeat(self.dataset_num)).apply(
                map_and_batch(img_data_class.image_processing,
                              batch_size=self.batch_size,
                              num_parallel_batches=16,
                              drop_remainder=True))

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

            if self.manager.latest_checkpoint:
                self.ckpt.restore(
                    self.manager.latest_checkpoint
                ).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

        if self.phase == 'valid':
            from khaiii import KhaiiiApi

            API = KhaiiiApi()

            n_max_words = 18
            all_captions = []
            with open(self.captions, 'r', encoding='euc-kr') as f:
                valid_captions = f.read().split('\n')
                self.dataset_num = len(valid_captions)

                for cap in valid_captions:
                    morphs = API.analyze(cap)
                    tokens = tokenizer(morphs)

                    all_captions.append(tokens)

            word_counts = defaultdict(float)
            for cap in all_captions:
                for word in cap:
                    word_counts[word] += 1

            vocab = [w for w in word_counts if word_counts[w] >= 0]

            idxtoword = {}
            idxtoword[0] = '<끝>'
            wordtoidx = {}
            wordtoidx['<끝>'] = 0
            idx = 1
            for word in vocab:
                wordtoidx[word] = idx
                idxtoword[idx] = word
                idx += 1

            self.captions_new = []
            for cap in all_captions:
                rev = []
                for word in cap:
                    if word in wordtoidx:
                        rev.append(wordtoidx[word])
                self.captions_new.append(rev)

            for i in range(self.dataset_num):
                self.captions_new[i] = self.captions_new[i][:n_max_words]
                self.captions_new[i] = self.captions_new[i] + [2] * \
                                       (n_max_words - len(self.captions_new[i]))

            self.captions_new = np.reshape(
                self.captions_new,
                [-1, self.dataset_num, n_max_words]
            )

            img_and_caption = data.Dataset.from_tensor_slices(self.captions_new)

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

            if self.manager.latest_checkpoint:
                self.ckpt.restore(
                    self.manager.latest_checkpoint
                ).expect_partial()
                print('Latest checkpoint restored!!')
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
                GradientTape() as d_256_tape:
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

            real_64 = image.resize(real_256, size=[64, 64],
                                   method=image.ResizeMethod.BILINEAR)
            real_128 = image.resize(real_256, size=[128, 128],
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

    def train(self):
        start_time = time.time()

        train_summary_writer = summary.create_file_writer(
            os.path.join(self.log_dir, self.model_dir)
        )

        for idx in range(self.start_iteration, self.iteration):
            if self.decay_flag:
                decay_start_step = self.decay_iter

                if idx > 0 and (idx % decay_start_step) == 0:
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                    self.g_optimizer.learning_rate = lr
                    for i in range(3):
                        self.d_optimizer[i].learning_rate = lr
                    self.embed_optimizer.learning_rate = lr

            real_256, caption, class_id = next(self.img_caption_iter)

            embed_loss = self.embed_train_step(real_256, caption, class_id)

            d_loss, d_adv_loss = self.d_train_step(real_256, caption)

            g_loss, g_adv_loss, g_kl_loss, g_embed_loss, fake_256 = \
                self.g_train_step(caption, class_id)
            g_loss += embed_loss

            with train_summary_writer.as_default():
                summary.scalar('g_adv_loss', g_adv_loss, step=idx)
                summary.scalar('g_kl_loss', g_kl_loss, step=idx)
                summary.scalar('g_embed_loss', g_embed_loss, step=idx)
                summary.scalar('g_loss', g_loss, step=idx)

                summary.scalar('embed_loss', embed_loss, step=idx)

                summary.scalar('d_adv_loss', d_adv_loss, step=idx)
                summary.scalar('d_loss', d_loss, step=idx)

            if np.mod(idx + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=idx + 1)

            if np.mod(idx + 1, self.print_freq) == 0:
                real_images = real_256[:5]
                fake_images = fake_256[:5]

                merge_real_images = np.expand_dims(
                    return_images(real_images, [5, 1]), axis=0
                )
                merge_fake_images = np.expand_dims(
                    return_images(fake_images, [5, 1]), axis=0
                )

                merge_images = np.concatenate(
                    [merge_real_images, merge_fake_images], axis=0
                )

                save_images(merge_images, [1, 2],
                            f'./{self.sample_dir}/merge_{idx+1:07d}.jpg')

            print(f'Iteration: [{idx:6d}/{self.iteration:6d}] '
                  f'time: {time.time() - start_time:4.4f} '
                  f'd_loss: {d_loss:.8f}, g_loss: {g_loss:8f}')

        self.manager.save(checkpoint_number=self.iteration)

    @property
    def model_dir(self):
        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return f'{self.model_name}_{self.gan_type}_{self.adv_weight}adv' \
               f'_{self.kl_weight}kl_{self.embed_weight}embed{sn}'

    def test(self):
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write('<html><body><table><tr>')
        index.write('<th>name</th><th>input</th><th>output</th></tr>')

        real_256, caption = next(self.img_caption_iter)
        target_sentence_index = random.uniform(shape=[], minval=0,
                                               maxval=10, dtype=int32)
        caption = gather(caption, target_sentence_index, axis=1)

        word_emb, sent_emb, mask = self.rnn_encoder(caption, training=False)

        z = random.normal(shape=[self.batch_size, self.z_dim])
        fake_64, fake_128, fake_256 = \
            self.generator([z, sent_emb, word_emb, mask], training=False)

        for i in range(5):
            real_path = os.path.join(self.result_dir, f'real_{i}.jpg')
            fake_path = os.path.join(self.result_dir, f'fake_{i}.jpg')

            real_image = np.expand_dims(real_256[i], axis=0)
            fake_image = np.expand_dims(fake_256[i], axis=0)

            save_images(real_image, [1, 1], real_path)
            save_images(fake_image, [1, 1], fake_path)

            index.write(f'<td>{os.path.basename(real_path)}</td>')
            index.write(f"<td><img src='{real_path if os.path.isabs(real_path) else ('../..' + os.path.sep + real_path)}' "
                        f"width='{self.img_width}' height='{self.img_height}'></td>")
            index.write(f"<td><img src='{fake_path if os.path.isabs(fake_path) else ('../..' + os.path.sep + fake_path)}' "
                        f"width='{self.img_width}' height='{self.img_height}'></td>")
            index.write(f'</tr>')

        index.close()