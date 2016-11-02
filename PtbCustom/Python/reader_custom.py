"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf

def read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
         sentences = f.read().decode("utf-8").replace("\n", "<eos>").split("<eos>")
         for i in xrange(len(sentences)):
             sentences[i] = sentences[i].split()
         return sentences
         
def prepare_sentence_training(sentences, max_length):
    sent = list(sentences)
    length = len(sent)
    for i in xrange(length):
        diff = max_length - len(sent[i])
        sent[i].insert(0,'<bos>')
        sent[i].append('<eos>')
        for j in xrange(diff+1):
            sent[i].append('@')
    return sent
         
def build_vocab(sentences):
    data = sentences
    data = [item for sublist in data for item in sublist]
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id
    
def calc_max_length(train_path,valid_path,test_path):
    train_sentences = read_sentences(train_path)
    valid_sentences = read_sentences(valid_path)
    test_sentences = read_sentences(test_path)
    max_length = 0
    train_length = len(train_sentences)
    for i in xrange(train_length):
        if len(train_sentences[i]) > max_length:
            max_length = len(train_sentences[i])         
    valid_length = len(valid_sentences)
    for i in xrange(valid_length):
        if len(valid_sentences[i]) > max_length:
            max_length = len(valid_sentences[i]) 
    test_length = len(test_sentences)
    for i in xrange(test_length):
        if len(test_sentences[i]) > max_length:
            max_length = len(test_sentences[i]) 
    return max_length
    
def file_to_word_ids(filename, word_to_id, max_length):
    data = read_sentences(filename)
    data = prepare_sentence_training(data, max_length)
    result = list(data)
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            result[i][j] = word_to_id[data[i][j]]
    return result

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    max_length = calc_max_length(train_path,valid_path,test_path)
    
    train_sentences = read_sentences(train_path)
    valid_sentences = read_sentences(valid_path)
    test_sentences = read_sentences(test_path)
    
    train_sentences = prepare_sentence_training(train_sentences, max_length)
    valid_sentences = prepare_sentence_training(valid_sentences, max_length)
    test_sentences = prepare_sentence_training(test_sentences, max_length)
    
    word_to_id = build_vocab(train_sentences)
    
    train_data = file_to_word_ids(train_path, word_to_id, max_length)
    valid_data = file_to_word_ids(valid_path, word_to_id, max_length)
    test_data = file_to_word_ids(test_path, word_to_id, max_length)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary, max_length+2
    
def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        nb_sentences = tf.shape(raw_data)[0]
        batch_nb_sentences = nb_sentences // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_nb_sentences],
                                           [batch_size, batch_nb_sentences*(num_steps+1)])

        epoch_size = (batch_nb_sentences)
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i*(num_steps+1)], [batch_size, num_steps])
        y = tf.slice(data, [0, i*(num_steps+1)+1], [batch_size, num_steps])
        return x, y
