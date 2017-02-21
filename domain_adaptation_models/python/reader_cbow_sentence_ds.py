from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
        try:
            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)

import tensorflow as tf
import numpy as np
import copy
from gensim import corpora, models

def read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
         sentences = f.read().decode("utf-8").split("\n")
         for i in xrange(len(sentences)):
             sentences[i] = sentences[i].split()
         return sentences
         
def build_vocab(sentences):
    data = sentences
    data = [item for sublist in data for item in sublist]
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id
    
# max_length should be a parameter that has to be tuned to gain efficiency, but we still have to take all test data into account
def calc_max_length(train_path,valid_path,test_path):
#    train_sentences = read_sentences(train_path)
#    valid_sentences = read_sentences(valid_path)
    test_sentences = read_sentences(test_path)
    max_length = 0
#    train_length = len(train_sentences)
#    for i in xrange(train_length):
#        if len(train_sentences[i]) > max_length:
#            max_length = len(train_sentences[i])         
#    valid_length = len(valid_sentences)
#    for i in xrange(valid_length):
#        if len(valid_sentences[i]) > max_length:
#            max_length = len(valid_sentences[i]) 
    test_length = len(test_sentences)
    for i in xrange(test_length):
        if len(test_sentences[i]) > max_length:
            max_length = len(test_sentences[i]) 
    return max_length

def file_to_word_ids(filename, word_to_id):
    data = read_sentences(filename)
    for i in xrange(len(data)):
        for j in xrange(len(data[i])):
            data[i][j] = word_to_id[data[i][j]]
    return data
    

def ds_raw_data(data_path):
    
    train_path = os.path.join(data_path, "ds.train.txt")
    valid_path = os.path.join(data_path, "ds.valid.txt")
    test_path = os.path.join(data_path, "ds.test.txt")
    
    max_length = calc_max_length(train_path,valid_path,test_path)
    
    #word_to_id = build_vocab(read_sentences(train_path))
    
    dict_path = os.path.join(data_path, "dictionary.ds.dict")
    dictionary = corpora.Dictionary.load(dict_path)
    
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
        word_to_id[word] = wordid
	
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    unk_id = word_to_id['<unk>']
    return train_data, valid_data, test_data, vocabulary, unk_id, max_length, word_to_id 
    
def ds_producer(raw_data, batch_size, max_length, num_history, word_to_id, name=None):

    data_length_array = np.zeros(len(raw_data),dtype=np.int32)
    for i in xrange(len(raw_data)):
        data_length_array[i] = min(len(raw_data[i]),max_length)
    average_length = np.mean(data_length_array)
        
    history = np.concatenate((len(word_to_id)*np.ones(num_history-1, dtype = np.int32),np.array([word for sentence in raw_data for word in sentence])))
    history_data = np.zeros([len(raw_data),max_length+num_history-1], dtype=np.int32)
    pos = 0
    for i in xrange(len(raw_data)):
        sentence_len = len(raw_data[i])
        if sentence_len <= max_length:
            history_data[i] = np.concatenate((history[pos:pos+sentence_len+num_history-1], len(word_to_id)*np.ones([max_length-sentence_len],dtype=np.int32)))
            pos += sentence_len
        else:
            history_data[i] = history[pos:pos+max_length+num_history-1]
            pos += sentence_len

    result = [[] for i in range(len(raw_data))]
    for i in xrange(len(raw_data)):
        for j in xrange(max_length+1):
            if j < len(raw_data[i]):
                result[i].append(raw_data[i][j])
            elif j == len(raw_data[i]):
                result[i].append(word_to_id['<bos>'])
            else:
                result[i].append(len(word_to_id))
    raw_data = list(result)
    
    with tf.name_scope(name):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        history_data = tf.convert_to_tensor(history_data, name="history_data", dtype=tf.int32)
        lengths = tf.convert_to_tensor(data_length_array, name="data_length_array", dtype=tf.int32)
        
        nb_sentences = tf.shape(raw_data)[0]
        nb_of_batches = nb_sentences // batch_size
        data = tf.reshape(raw_data[0 : batch_size * nb_of_batches],
                                           [batch_size, nb_of_batches*(max_length+1)])
        history_data = tf.reshape(history_data[0 : batch_size * nb_of_batches], 
                                            [batch_size, nb_of_batches*(max_length+num_history-1)])
        lengths1 = tf.reshape(lengths[0 : batch_size * nb_of_batches],
                                             [batch_size, nb_of_batches])
        lengths2 = tf.reduce_max(lengths1,reduction_indices=0)
        #lengths3 = tf.add(lengths2, [num_history-1])

        epoch_size = (nb_of_batches)

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i*(max_length+1)], [batch_size, max_length]) #tf.squeeze(tf.slice(lengths2,[i],[1]))])
        y = tf.slice(data, [0, i*(max_length+1)+1], [batch_size, max_length]) #tf.squeeze(tf.slice(lengths2,[i],[1]))])
        z = tf.slice(history_data, [0, i*(max_length+num_history-1)], [batch_size, max_length + num_history -1]) #tf.squeeze(tf.slice(lengths3,[i],[1]))])
        
        return x, y, tf.slice(lengths2,[i],[1]), z, average_length
