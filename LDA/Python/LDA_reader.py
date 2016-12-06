
##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np

##### topic parameters
nb_topics = 100

##### functions

def _read_words_no_split(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>")

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

class MyCorpus(object):
    def __init__(self,texts,dictionary):
        self.corpus = texts
        self.dict = dictionary
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.dict.doc2bow(self.corpus[i])

def ptb_raw_data_lda(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
    data_path: string path to the directory where simple-examples.tgz has
    been extracted.

    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    doctrain = _read_words_no_split('/home/wim/SpeechHub/LDA/Input/ptb.train.txt')
    docvalid = _read_words_no_split('/home/wim/SpeechHub/LDA/Input/ptb.valid.txt')
 
    docs = [doctrain,docvalid]
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.Dictionary(texts)

    corpus = MyCorpus(texts,dictionary)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=nb_topics)
    
    #lda.save('lda.model')
    #dictionary.save('dict.dictionary')
    
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
        word_to_id[word] = wordid
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    
    topic_array = np.zeros((nb_topics, vocabulary))
    for topic_nb in xrange(nb_topics):
        current_topic = lda.get_topic_terms(topic_nb,topn=vocabulary)
        for i in xrange(vocabulary):
            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]

    

    return train_data, valid_data, test_data, vocabulary, topic_array


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        num_steps: int, the number of unrolls.
        name: the name of this operation (optional).

    Returns:
        A pair of Tensors, each shaped [batch_size, num_steps]. The second element
        of the tuple is the same data time-shifted to the right by one.

    Raises:
        tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
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
