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


def get_targets(words, idx, window_size=5):
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])
    return list(targets)

def get_batches(words, batch_size, window_size=5):
    pass


inputs = tf.placholder(tf.int32, shape=[None], name='inputs')
outputs = tf.placeholder(tf.int32, shape=[None, None], name='labels')
embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
embed = tf.nn.embedding_lookup(embedding, inputs)

n_sampled = 100
softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
softmax_b = tf.Variable(tf.zeros(vocab_sieze))

loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer().minimize(cost)

train_grapg = tf.Graph()
with train_graph.as_default():
    valid_size = 7
    valid_window = 100
    valid_examples = np.array(random.sample(range(valid_window), valid_sieze//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))

    valid_examples = [vocab_to_int['word'],
                      vocab_to_int['ppt'],
                      vocab_to_int['熟悉'],
                      vocab_to_int['java'],
                      vocab_to_int['能力'],
                      vocab_to_int['逻辑思维'],
                      vocab_to_int['了解']]
	valid_size = len(valid_examples)
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
	normalized_embedding = embedding / norm
	valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
	similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


epochsepochs  ==  1010  # 迭代轮数# 迭代轮数
 batch_sizebatch_si  = 1000 # batch大小
window_size = 10 # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver() # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        # 
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss/100),
                      "{:.4f} sec/batch".format((end-start)/100))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8 # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)
