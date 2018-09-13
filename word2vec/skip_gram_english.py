#!/usr/bin/env python3
#-*- coding: utf-8 -*-


import time
import random
import logging
from collections import Counter

import numpy as np
import tensorflow as tf

CORPUS = './data/text8.txt'

logging.basicConfig(level=logging.DEBUG,
                    datefmt='%m-%d %H:%M:%S',
                    format='[%(asctime)s.%(msecs)03d] [%(levelname)-4s] %(message)s')
logger = logging.getLogger(__name__)

def read_data(data_path):
    with open(data_path, encoding='utf-8') as f:
        data = f.read()
    return data


def preprocess(text, freq=5):
    '''
    对文本进行预处理

    参数
    ---
    text: 文本数据
    freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]

    return trimmed_words

words = preprocess(read_data(CORPUS))

vocab = set(words)
#print(words[:20])

vocab_to_int = { w: i for i, w in enumerate(words) }
int_to_vocab = { i: w for i, w in enumerate(words) }

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))

int_words = [ vocab_to_int[w] for w in words ]


# negative sampling
t = 1e-5
threshold = 0.8

int_word_counts = Counter(int_words)
total_count = len(int_word_counts)

word_freqs =  { w: c/total_counts for w, c in int_word_counts.items() }
prob_drop = { w: 1 - np.sqrt(t/word_freqs(w)) for w in int_word_counts }
train_words = [ w for w in int_words if prob_drop[w] < threshold ]

print(len(train_words))
