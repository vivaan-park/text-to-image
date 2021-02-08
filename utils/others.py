# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from tensorflow import config

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def str2bool(x):
    return x.lower() in ('true')

def automatic_gpu_usage() :
    gpus = config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                config.experimental.set_memory_growth(gpu, True)
            logical_gpus = config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)