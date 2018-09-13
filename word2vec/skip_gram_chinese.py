#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import time
import random
from collections import Counter

import numpy as np
import tensorflow as tf


def read_data(data_path):
    with open(data_path, encoding='utf-8') as f:
        data = f.read()
    return data


# 数据预处理
# 1. 替换文本中特殊符号并去除低频词
# 2. 对文本分词
# 3. 构建语料
# 4. 单词映射表
word_count = Counter(words)

