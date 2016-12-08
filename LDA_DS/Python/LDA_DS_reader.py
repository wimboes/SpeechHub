##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np
import string

filename = '/home/wim/SpeechHub/LDA_DS/Input/test.txt'

##### parameters

train_split_nb = 100

##### functions

def read_words_split(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("<s>", "").replace("</s>","").replace("\n", "").split()
        
def read_and_split_doc(filename,split_nb):
    with tf.gfile.GFile(filename, "r") as f:
        full_doc = f.read().decode("utf-8").replace("<s>", "").replace("</s>","").replace("\n", "").split()  
    len_full_doc = len(full_doc)
    wpsd = int(np.floor(len_full_doc/split_nb))# words per small document
    split_doc = []
    for i in xrange(split_nb):
        split_doc.append(string.join(full_doc[i*wpsd:(i+1)*wpsd],sep=" "))         
    return split_doc

#def _file_to_word_ids(filename, word_to_id):
#    data = _read_words(filename)
#    return [word_to_id[word] for word in data if word in word_to_id]
#
#def ptb_raw_data_lda(data_path=None,lda_path='lda.model',dict_path='dictionary.dict'):
#    train_path = os.path.join(data_path, "ptb.train.txt")
#    valid_path = os.path.join(data_path, "ptb.valid.txt")
#    test_path = os.path.join(data_path, "ptb.test.txt")
#    
#    lda = models.LdaModel.load(lda_path)
#    dictionary = corpora.Dictionary.load(dict_path)
#    
#    word_to_id = dict()
#    for (wordid,word) in dictionary.iteritems():
#        word_to_id[word] = wordid
#    train_data = _file_to_word_ids(train_path, word_to_id)
#    valid_data = _file_to_word_ids(valid_path, word_to_id)
#    test_data = _file_to_word_ids(test_path, word_to_id)
#    vocabulary = len(word_to_id)
#    
#    topic_array = np.zeros((nb_topics, vocabulary))
#    for topic_nb in xrange(nb_topics):
#        current_topic = lda.get_topic_terms(topic_nb,topn=vocabulary)
#        for i in xrange(vocabulary):
#            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]
#
#    return train_data, valid_data, test_data, vocabulary, topic_array
#
#
#def ptb_producer(raw_data, batch_size, num_steps, name=None):
#    """Iterate on the raw PTB data.
#
#    This chunks up raw_data into batches of examples and returns Tensors that
#    are drawn from these batches.
#
#    Args:
#        raw_data: one of the raw data outputs from ptb_raw_data.
#        batch_size: int, the batch size.
#        num_steps: int, the number of unrolls.
#        name: the name of this operation (optional).
#
#    Returns:
#        A pair of Tensors, each shaped [batch_size, num_steps]. The second element
#        of the tuple is the same data time-shifted to the right by one.
#
#    Raises:
#        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
#    """
#    with tf.name_scope(name):
#        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
#
#        data_len = tf.size(raw_data)
#        batch_len = data_len // batch_size
#        data = tf.reshape(raw_data[0 : batch_size * batch_len],
#                          [batch_size, batch_len])
#
#        epoch_size = (batch_len - 1) // num_steps
#        assertion = tf.assert_positive(
#                                       epoch_size,
#                                       message="epoch_size == 0, decrease batch_size or num_steps")
#        with tf.control_dependencies([assertion]):
#            epoch_size = tf.identity(epoch_size, name="epoch_size")
#
#        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
#        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
#        return x, y
