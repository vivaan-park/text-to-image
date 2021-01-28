# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

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