# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os
import pickle
from collections import defaultdict

from khaiii import KhaiiiApi

API = KhaiiiApi()

CAPTIONS_PER_IMAGE = 10

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

def build_dictionary(train_captions, test_captions):
    word_counts = defaultdict(float)
    captions = train_captions + test_captions
    for cap in captions:
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

    train_captions_new = []
    for cap in train_captions:
        rev = []
        for word in cap:
            if word in wordtoidx:
                rev.append(wordtoidx[word])
        train_captions_new.append(rev)

    test_captions_new = []
    for cap in test_captions:
        rev = []
        for word in cap:
            if word in wordtoidx:
                rev.append(wordtoidx[word])
        test_captions_new.append(rev)

    return [train_captions_new, test_captions_new,
            idxtoword, wordtoidx, len(idxtoword)]
