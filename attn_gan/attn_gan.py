# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from utils.others import check_folder

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