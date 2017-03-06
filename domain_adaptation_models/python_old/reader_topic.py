##### comments

# first run lda_lm_ds.py and regular_lm_ds.py before running this file
# combineren lda en gewoon taalmodel

##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np
import copy

##### functions

def read_sentences(path):
    with tf.gfile.GFile(path, "r") as f:
        sentences = f.read().decode("ascii", 'ignore').split("\n")
    for i in xrange(len(sentences)):
        sentences[i] = sentences[i].split()
    return sentences

def file_to_word_ids(path, word_to_id, max_length):
    sentences = read_sentences(path)
    result = copy.deepcopy(sentences)
    vocab_size = len(word_to_id)
    for i in xrange(len(sentences)):
        for j in xrange(max_length+1):
            if j < len(sentences[i]):
                result[i][j] = word_to_id[sentences[i][j]]
            elif j == len(sentences[i]):
                result[i].append(word_to_id['<bos>'])
            else:
                result[i].append(vocab_size)
    return result
    
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
  
def ds_raw_data_lda(data_path):
    
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    
    max_length = calc_max_length(train_path,valid_path,test_path)
    
    lda_path = os.path.join(data_path, "lda.model")
    dict_path = os.path.join(data_path, "dictionary.dict")
    
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
        train_length_array[i] = len(train_sentences[i])

    valid_sentences = read_sentences(valid_path)
    valid_length_array = np.zeros(len(valid_sentences),dtype=np.int32)
    for i in xrange(len(valid_sentences)):
        valid_length_array[i] = len(valid_sentences[i])

    test_sentences = read_sentences(test_path)
    test_length_array = np.zeros(len(test_sentences),dtype=np.int32)
    for i in xrange(len(test_sentences)):
        test_length_array[i] = len(test_sentences[i])

    return train_data, valid_data, test_data, vocab_size, unk_id, max_length, topic_array, train_length_array, valid_length_array, test_length_array

def ds_producer(raw_data, batch_size, length_array, max_length, name=None):

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
        x = tf.slice(data, [0, i*(max_length+1)], [batch_size, tf.squeeze(tf.slice(lengths2,[i],[1]))])
        y = tf.slice(data, [0, i*(max_length+1)+1], [batch_size, tf.squeeze(tf.slice(lengths2,[i],[1]))])
        
        return x, y, tf.slice(lengths2,[i],[1])