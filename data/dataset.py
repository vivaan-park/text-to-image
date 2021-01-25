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

def tokenizer(cap):
    tokens = []
    for word in cap:
        for morphs in word.morphs:
            w, m = str(morphs).split('/')
            if m not in ('SF', 'SP', 'SS', 'SE', 'SO', 'SW'):
                tokens.append(w)
    return tokens

def load_captions(filenames):
    all_captions = []
    for i in range(len(filenames)):
        cap_path = f'texts/{filenames[i]}.txt'
        with open(cap_path, 'r') as f:
            captions = f.read().split('\n')
            cnt = 0
            for cap in captions:
                if not len(cap):
                    continue

                morphs = API.analyze(cap.lower())
                tokens = tokenizer(morphs)
                if not len(tokens):
                    continue

                all_captions.append(tokens)
                cnt += 1
                if cnt == CAPTIONS_PER_IMAGE:
                    break
            if cnt < CAPTIONS_PER_IMAGE:
                print(f'ERROR: the captions for {filenames[i]} less than {cnt}')
    return all_captions
