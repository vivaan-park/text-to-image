# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import argparse

from utils.others import check_folder, str2bool
from attn_gan import AttnGAN

def check_args(args):
    check_folder(args.checkpoint_dir)
    check_folder(args.result_dir)
    check_folder(args.log_dir)
    check_folder(args.sample_dir)

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train',
                        choices=('train', 'test'), help='phase name')

    parser.add_argument('--iteration', type=int, default=1000000,
                        help='The number of training iterations')
    parser.add_argument('--decay_flag', type=str2bool, default=True,
                        help='The decay_flag')
    parser.add_argument('--decay_iter', type=int, default=500000,
                        help='decay epoch')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='The size of batch size for each gpu')
    parser.add_argument('--print_freq', type=int, default=1000,
                        help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='The number of ckpt_save_freq')

    parser.add_argument('--lr', type=float, default=0.0002,
                        help='The learning rate')

    parser.add_argument('--gan_type', type=str, default='gan',
                        help='[gan / lsgan / hinge]')

    parser.add_argument('--adv_weight', type=int, default=1,
                        help='Weight about GAN')
    parser.add_argument('--kl_weight', type=int, default=1,
                        help='Weight about kl_loss')
    parser.add_argument('--embed_weight', type=int, default=1,
                        help='Weight about embed_weight')

    parser.add_argument('--z_dim', type=int, default=256,
                        help='condition & noise z dimension')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='embedding dimension')
    parser.add_argument('--g_dim', type=int, default=32,
                        help='generator feature basic dimension')
    parser.add_argument('--d_dim', type=int, default=64,
                        help='discriminaotr feature basic dimension')

    parser.add_argument('--sn', type=str2bool, default=False,
                        help='using spectral norm')

    parser.add_argument('--img_height', type=int, default=256,
                        help='The height size of image')
    parser.add_argument('--img_width', type=int, default=256,
                        help='The width size of image ')
    parser.add_argument('--img_ch', type=int, default=3,
                        help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=False,
                        help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--valid_dir', type=str, default='validation',
                        help='Directory name to save the validation')
    parser.add_argument('--captions', type=str, default='captions.txt',
                        help='Captions file path to validate')

    return check_args(parser.parse_args())

def main():
    args = parse_args()

    gan = AttnGAN(args)
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(' [*] Training finished!')

    if args.phase == 'test':
        gan.test()
        print(' [*] Test finished!')


if __name__ == '__main__':
    main()