##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
from gensim import corpora, models
import numpy as np
import copy

##### functions

def read_sentences(path):
    with tf.gfile.GFile(path, "r") as f:
        sentences = f.read().decode("utf-8").split("\n")
    for i in xrange(len(sentences)):
        sentences[i] = sentences[i].split()
    return sentences

def file_to_word_ids(path, word_to_id, max_length):
    sentences = read_sentences(path)
    result = np.zeros((len(sentences),max_length+1))
    vocab_size = len(word_to_id)
    for i in xrange(len(sentences)):
        for j in xrange(max_length+1):
            if j < len(sentences[i]):
                result[i][j] = word_to_id[sentences[i][j]]
            elif j == len(sentences[i]):
                result[i][j] = word_to_id['<bos>']
            else:
                result[i][j] = vocab_size
    return result
    
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
  
def ptb_raw_data(data_path):
    
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    max_length = calc_max_length(train_path,valid_path,test_path)
    
    lda_path = os.path.join(data_path, "lda.ptb.model")
    dict_path = os.path.join(data_path, "dictionary.ptb.dict")
    
    lda = models.LdaModel.load(lda_path)
    dictionary = corpora.Dictionary.load(dict_path)
    
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
        word_to_id[word] = wordid
    unk_id = word_to_id['<unk>']
    train_data = file_to_word_ids(train_path, word_to_id, max_length)
    valid_data = file_to_word_ids(valid_path, word_to_id, max_length)
    test_data = file_to_word_ids(test_path, word_to_id, max_length)
    vocab_size = len(word_to_id)
    
    nb_topics = lda.num_topics
    topic_array = np.zeros((nb_topics, vocab_size))
    for topic_nb in xrange(nb_topics):
        current_topic = lda.get_topic_terms(topic_nb,topn=vocab_size)
        for i in xrange(vocab_size):
            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]

    train_sentences = read_sentences(train_path)
    train_length_array = np.zeros(len(train_sentences),dtype=np.int32)
    for i in xrange(len(train_sentences)):
        train_length_array[i] = min(len(train_sentences[i]),max_length)

    valid_sentences = read_sentences(valid_path)
    valid_length_array = np.zeros(len(valid_sentences),dtype=np.int32)
    for i in xrange(len(valid_sentences)):
        valid_length_array[i] = min(len(valid_sentences[i]),max_length)

    test_sentences = read_sentences(test_path)
    test_length_array = np.zeros(len(test_sentences),dtype=np.int32)
    for i in xrange(len(test_sentences)):
        test_length_array[i] = min(len(test_sentences[i]),max_length)

    return train_data, valid_data, test_data, vocab_size, unk_id, max_length, topic_array, train_length_array, valid_length_array, test_length_array

def ptb_producer(raw_data, batch_size, length_array, max_length, name=None):
    
#    print(max_length)
#    print(len(raw_data))
#    print(len(raw_data[0]))
#    print(raw_data[4607])
#    print(len(raw_data[4607]))
#    print(raw_data[4608])
#    print(len(raw_data[4608]))
#    print(raw_data[4609])
#    words = [words for zin in raw_data for words in zin]
#    print(words.index('<unk>'))
#    print(words[360000:370000])

    with tf.name_scope(name):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        nb_sentences = tf.shape(raw_data)[0]
        nb_batches = nb_sentences // batch_size
        data = tf.reshape(raw_data[0 : batch_size * nb_batches],
                                           [batch_size, nb_batches*(max_length+1)])
        
        lengths = tf.convert_to_tensor(length_array, name="length_array", dtype=tf.int32)
        lengths1 = tf.reshape(lengths[0 : batch_size * nb_batches],
                                             [batch_size, nb_batches])
        lengths2 = tf.reduce_max(lengths1,reduction_indices=0)

        epoch_size = (nb_batches)

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        tf.slice(lengths2,[i],[1])
        x = tf.slice(data, [0, i*(max_length+1)], [batch_size,
 tf.squeeze(tf.slice(lengths2,[i],[1]))])
        y = tf.slice(data, [0, i*(max_length+1)+1], [batch_size, tf.squeeze(tf.slice(lengths2,[i],[1]))])
        
        return x, y, tf.slice(lengths2,[i],[1])

