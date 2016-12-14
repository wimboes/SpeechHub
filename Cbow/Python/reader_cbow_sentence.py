#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:22:42 2016

@author: robbe & wim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np

def read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
         sentences = f.read().decode("ascii", 'ignore').replace("\n", "<eos>").split("<eos>")
         for i in xrange(len(sentences)):
             sentences[i] = sentences[i].split()
         return sentences[0:-1]
         
def prepare_sentence_training(sentences):
    sent = list(sentences)
    for i in xrange(len(sent)):
        sent[i].insert(0,'<bos>')
        sent[i].append('<eos>')
    return sent
         
def build_vocab(sentences):
    data = sentences
    data = [item for sublist in data for item in sublist]
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1,1+len(words))))

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

def file_to_word_ids(filename, word_to_id):
    data = read_sentences(filename)
    data = prepare_sentence_training(data)
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            data[i][j] = word_to_id[data[i][j]]
    return data
    

def ptb_raw_data(data_path, text_data, vocab_size):
    
    if text_data == 'PTB' :
        train_path = os.path.join(data_path, "ptb.train.txt")
        valid_path = os.path.join(data_path, "ptb.valid.txt")
        test_path = os.path.join(data_path, "ptb.test.txt")
    elif text_data == 'DS' :
        train_path = os.path.join(data_path, "ds.train"+ str(vocab_size) +".txt")
        valid_path = os.path.join(data_path, "ds.valid"+ str(vocab_size) +".txt")
        test_path = os.path.join(data_path, "ds.test"+ str(vocab_size) +".txt")
    max_length = calc_max_length(train_path,valid_path,test_path)

    train_sentences = prepare_sentence_training(read_sentences(train_path))
    
    word_to_id = build_vocab(train_sentences)
    
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    unk_id = word_to_id['<unk>']
    return train_data, valid_data, test_data, vocabulary, unk_id, max_length+2 
    
def ptb_producer(raw_data, batch_size, num_steps,num_history, name=None):

    data_length_array = np.zeros(len(raw_data),dtype=np.int32)
    for i in xrange(len(raw_data)):
        data_length_array[i] = len(raw_data[i])-1

    history = np.concatenate((np.zeros(num_history-1, dtype = np.int32),np.array([word for sentence in raw_data for word in sentence])))
    history_data = np.zeros([len(raw_data),num_steps+num_history-1], dtype=np.int32)
    pos = 0
    for i in xrange(len(raw_data)):
        sentence_len = len(raw_data[i])
        history_data[i] = np.concatenate((history[pos:pos+sentence_len+num_history-1], np.zeros([num_steps-sentence_len],dtype=np.int32)))
        pos += sentence_len

    result = list(raw_data)
    for i in xrange(len(raw_data)):
        for j in xrange(num_steps+1):
            if j < len(raw_data[i]):
                result[i][j] = raw_data[i][j]
            else:
                result[i].append(0)
    raw_data = list(result)
    
    with tf.name_scope(name):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        history_data = tf.convert_to_tensor(history_data, name="history_data", dtype=tf.int32)
        lengths = tf.convert_to_tensor(data_length_array, name="data_length_array", dtype=tf.int32)
        
        nb_sentences = tf.shape(raw_data)[0]
        nb_of_batches = nb_sentences // batch_size
        data = tf.reshape(raw_data[0 : batch_size * nb_of_batches],
                                           [batch_size, nb_of_batches*(num_steps+1)])
        history_data = tf.reshape(history_data[0 : batch_size * nb_of_batches], 
                                            [batch_size, nb_of_batches*(num_steps+num_history-1)])
        lengths1 = tf.reshape(lengths[0 : batch_size * nb_of_batches],
                                             [batch_size, nb_of_batches])
        lengths2 = tf.reduce_max(lengths1,reduction_indices=0)
        lengths3 = tf.add(lengths2, [num_history-1])

        epoch_size = (nb_of_batches)

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i*(num_steps+1)], [batch_size, num_steps])#tf.squeeze(tf.slice(lengths2,[i],[1]))])
        y = tf.slice(data, [0, i*(num_steps+1)+1], [batch_size, num_steps])#tf.squeeze(tf.slice(lengths2,[i],[1]))])
        z = tf.slice(history_data, [0, i*(num_steps+num_history-1)], [batch_size, num_steps + num_history -1])#tf.squeeze(tf.slice(lengths3,[i],[1]))])
        
        return x, y, tf.slice(lengths2,[i],[1]), z
        
python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
data_path = os.path.join(general_path,'Input')

batch_size = 3
num_history = 10

a  = ptb_raw_data(data_path, 'PTB', 10000)
train_data, valid_data, test_data, vocabulary, unk_id, num_steps = a
x,y, num, z = ptb_producer(train_data, batch_size, num_steps, num_history)

print('ok')

sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess, coord=coord)

[l,p,q,r] = sess.run([x,y,num,z])
print(l)
print(p)
print(q)
print(r)
print('ok2 \n')
[l,p,q,r] = sess.run([x,y,num,z])
print(l)
print(p)
print(q)
print(r)
print('ok3 \n')
[l,p,q,r] = sess.run([x,y,num,z])
print(l)
print(p)
print(q)
print(r)
for i in range(100):
    [l,p,q,r] = sess.run([x,y,num,z])
[l,p,q,r] = sess.run([x,y,num,z])
print(l)
print(p)
print(q)
print(r)