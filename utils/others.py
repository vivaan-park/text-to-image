# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')