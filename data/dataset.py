# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import pickle

def load_filenames(data_dir, split):
    filepath = f'{data_dir}/filenames_{split}.pickle'
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print(f'Load {split} filenames ({len(filenames)})')
    else:
        filenames = []
    return filenames
