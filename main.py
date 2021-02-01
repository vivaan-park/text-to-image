# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from utils.others import check_folder

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
