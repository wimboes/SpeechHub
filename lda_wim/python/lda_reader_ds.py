##### comments

# first run transform_ds.py and lda_generator_ds.py before running this file
# nog testen

##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np

##### functions

def read_words(path):
    with tf.gfile.GFile(path, "r") as f:
        return f.read().decode("ascii", 'ignore').replace("\n"," ").split()

def file_to_word_ids(path, word_to_id):
    words = read_words(path)
    return [word_to_id[word] for word in words if word in word_to_id]
  
def lda_raw_data_ds(data_path):
    
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    
    lda_path = os.path.join(data_path, "lda.model")
    dict_path = os.path.join(data_path, "dictionary.dict")
    
    lda = models.LdaModel.load(lda_path)
    dictionary = corpora.Dictionary.load(dict_path)
    
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
        word_to_id[word] = wordid
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocab_size = len(word_to_id)
    
    nb_topics = lda.num_topics
    topic_array = np.zeros((nb_topics, vocab_size))
    for topic_nb in xrange(nb_topics):
        current_topic = lda.get_topic_terms(topic_nb,topn=vocab_size)
        for i in xrange(vocab_size):
            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]

    return train_data, valid_data, test_data, vocab_size, topic_array

def lda_producer_ds(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
                                       epoch_size,
                                       message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
        return x, y