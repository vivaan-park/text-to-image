# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import random
import pickle

import tensorflow as tf
import numpy as np

class Image_data:
    def __init__(self, img_height, img_width, channels,
                 dataset_path, augment_flag):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, 'images')
        self.text_path = os.path.join(dataset_path, 'texts')
        self.pickle_path = os.path.join(dataset_path, 'pickles')

        self.train_image_filename_pickle = \
            os.path.join(self.pickle_path, 'filenames_train.pickle')
        self.test_image_filename_pickle = \
            os.path.join(self.pickle_path, 'filenames_test.pickle')
        self.caption_pickle = \
            os.path.join(self.text_path, 'captions.pickle')
        self.class_info_pickle = \
            os.path.join(self.text_path, 'class_info.pickle')

    def image_processing(self, filename, captions, class_id=None):
        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels,
                                        dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag:
            augment_height_size = self.img_height + \
                                  (30 if self.img_height == 256
                                   else int(self.img_height * 0.1))
            augment_width_size = self.img_width + \
                                 (30 if self.img_width == 256
                                  else int(self.img_width * 0.1))

            seed = random.randint(0, 2 ** 31 - 1)
            condition = tf.greater_equal(
                tf.random.uniform(shape=[], minval=0.0, maxval=1.0),
                0.5
            )

            img = tf.cond(pred=condition,
                          true_fn=lambda: augmentation(
                              img, augment_height_size,
                              augment_width_size, seed
                          ),
                          false_fn=lambda: img)

        return img, captions, class_id

    def preprocess(self):
        train_images = []
        test_images = []

        with open(self.caption_pickle, 'rb') as f:
            x = pickle.load(f)

            train_captions = x[0]
            test_captions = x[1]
            n_max_words = 18

            train_captions = pad_sequence(train_captions,
                                          n_max_words, mode='post')
            test_captions = pad_sequence(test_captions,
                                         n_max_words, mode='post')

            idx_to_word = x[2]
            word_to_idx = x[3]

        with open(self.train_image_filename_pickle, 'rb') as f:
            x_list = pickle.load(f)

            for x in x_list :
                folder_name = x.split('/')[0]
                file_name = x.split('/')[1] + '.jpg'

                train_images.append(os.path.join(self.image_path,
                                                 folder_name, file_name))

        with open(self.test_image_filename_pickle, 'rb') as f:
            x_list = pickle.load(f)

            for x in x_list :
                folder_name = x.split('/')[0]
                file_name = x.split('/')[1] + '.jpg'

                test_images.append(os.path.join(self.image_path,
                                                folder_name, file_name))

        with open(self.class_info_pickle, 'rb') as f:
            class_id = pickle.load(f, encoding='latin1')

        return class_id, train_captions, train_images,\
               test_captions, test_images, idx_to_word, word_to_idx

def augmentation(image, augment_height, augment_width, seed):
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, [augment_height, augment_width])
    image = tf.image.random_crop(image, ori_image_shape, seed=seed)
    return image

def pad_sequence(captions, n_max_words, mode='post') :
    if mode == 'post':
        for i in range(len(captions)):
            captions[i] = captions[i][:n_max_words]
            captions[i] = captions[i] + [2] * (n_max_words - len(captions[i]))
    else:
        for i in range(len(captions)):
            captions[i] = captions[i][:n_max_words]
            captions[i] = [2] * (n_max_words - len(captions[i])) + captions[i]

    captions = np.reshape(captions, [-1, 10, n_max_words])
    return captions