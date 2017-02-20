##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

#if 'LD_LIBRARY_PATH' not in os.environ:
#        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
#        try:
#            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
#                sys.exit(0)
#        except Exception, exc:
#                print('Failed re_exec:', exc)
#                sys.exit(1)

import tensorflow as tf
from gensim import corpora, models
import numpy as np
import fnmatch

##### functions
        
class ds_data(object):
    def __init__(self, batch_size, data_path, name):
        
        #reading data
        path = os.path.join(data_path, name)
        self.amount_sentences = sum(1 for line in open(path))
        self._epoch_size = self.amount_sentences//batch_size
        self.amount_sentences = self.epoch_size*batch_size
        self.batch_size = batch_size
        
        #creating batch_files that will later be read
        self.directory = os.path.join(data_path, name + '_batch_files')
        if not os.path.exists(self.directory):
            print("Creating directory %s" % self.directory)
            os.mkdir(self.directory)
        if not len(fnmatch.filter(os.listdir(self.directory), '*.txt')) == batch_size:
            filelist = [ f for f in os.listdir(self.directory)]
            for f in filelist:
                os.remove(os.path.join(self.directory,f))
            with open(path, "r") as f:
                for i in xrange(batch_size):
                    batch_file = os.path.join(self.directory, 'batch' + str(i) + '.txt')
                    with open(batch_file, 'w') as bf:
                        for j in xrange(self._epoch_size):
                            bf.write(f.readline().decode("utf-8").encode('utf-8'))
            print('creating batch_files done')
            
        #reading word_to_id
        dict_path = os.path.join(data_path, "dictionary.ds.dict")
        dictionary = corpora.Dictionary.load(dict_path)
        self._word_to_id = dict()
        for (wordid,word) in dictionary.iteritems():
            self._word_to_id[word] = wordid
        self._unk_id = self._word_to_id['<unk>']
        self._bos_id = self._word_to_id['<bos>']
        self._eos_id = self._word_to_id['<eos>']
        self._pad_id = len(self._word_to_id)
        
        self.batch_id = 0
        

    def next_batch(self, max_seq_len):
        if self.batch_id == self.epoch_size:
            self.batch_id = 0
        batch_data = self.pad_id*np.ones((self.batch_size,max_seq_len))
        batch_labels = self.pad_id*np.ones((self.batch_size,max_seq_len))
        seq_len = np.zeros(self.batch_size)
        
        for i in xrange(self.batch_size):
            with open(os.path.join(self.directory, 'batch' + str(i) + '.txt'), 'r') as f:
                new_sentence = [lines.decode('utf-8').split() for lines in f][self.batch_id]  #niet effecient, moet beter voor train.data
            new_sentence = [self._word_to_id[word] for word in new_sentence]
            seqlen = len(new_sentence)
            seq_len[i] = min(seqlen, max_seq_len)
            
            batch_data[i,0:min(seqlen, max_seq_len)] = new_sentence[0:min(seqlen, max_seq_len)]
            new_sentence.append(self.bos_id)
            batch_labels[i,0:min(seqlen, max_seq_len)] = new_sentence[1:min(seqlen, max_seq_len)+1]
        self.batch_id += 1 
        return batch_data, batch_labels, seq_len
        
    def print_next_batch(self, max_seq_len):
        batch_data, batch_labels, seq_len = self.next_batch(max_seq_len)
        reverse = {v: k for k, v in self._word_to_id.iteritems()}
        reverse[self.pad_id] = 'PAD'
        batch_lst = [reverse[int(word_id)] for sentence in batch_data for word_id in sentence]
        targets_lst = [reverse[int(word_id)] for sentence in batch_labels for word_id in sentence]
        for i in xrange(self.batch_size):
            print('batch ' + str(i))
            print(batch_lst[max_seq_len*i:max_seq_len*(i+1)])
            print(targets_lst[max_seq_len*i:max_seq_len*(i+1)])
            print(seq_len[i])
        
    @property
    def unk_id(self):
        return self._unk_id
        
    @property
    def bos_id(self):
        return self._bos_id
        
    @property
    def eos_id(self):
        return self._eos_id
        
    @property
    def pad_id(self):
        return self._pad_id

    @property
    def word_to_id(self):
        return self._word_to_id
        
    @property
    def epoch_size(self):
        return self._epoch_size

class ds_data_with_history(object):
    def __init__(self, batch_size, history_size, data_path, name):
        
        #reading data
        path = os.path.join(data_path, name)
        self.amount_sentences = sum(1 for line in open(path))
        self._epoch_size = self.amount_sentences//batch_size
        self.amount_sentences = self.epoch_size*batch_size
        self.batch_size = batch_size
        self.history_size = history_size
        
        #creating batch_files that will later be read
        self.directory = os.path.join(data_path, name + '_batch_files')
        if not os.path.exists(self.directory):
            print("Creating directory %s" % self.directory)
            os.mkdir(self.directory)
        if not len(fnmatch.filter(os.listdir(self.directory), '*.txt')) == batch_size:
            filelist = [ f for f in os.listdir(self.directory)]
            for f in filelist:
                os.remove(os.path.join(self.directory,f))
            with open(path, "r") as f:
                for i in xrange(batch_size):
                    batch_file = os.path.join(self.directory, 'batch' + str(i) + '.txt')
                    with open(batch_file, 'w') as bf:
                        for j in xrange(self._epoch_size):
                            bf.write(f.readline().decode("utf-8").encode('utf-8'))
            print('creating batch_files done')
            
        #reading word_to_id
        dict_path = os.path.join(data_path, "dictionary.ds.dict")
        dictionary = corpora.Dictionary.load(dict_path)
        self._word_to_id = dict()
        for (wordid,word) in dictionary.iteritems():
            self._word_to_id[word] = wordid
        self._unk_id = self._word_to_id['<unk>']
        self._bos_id = self._word_to_id['<bos>']
        self._eos_id = self._word_to_id['<eos>']
        self._pad_id = len(self._word_to_id)
        
        self.batch_id = 0
        self.history = self.pad_id*np.ones((batch_size,history_size))
        

    def next_batch(self, max_seq_len):
        if self.batch_id == self.epoch_size:
            self.batch_id = 0
        batch_data = self.pad_id*np.ones((self.batch_size,max_seq_len))
        history_data = np.concatenate((self.history,self.pad_id*np.ones((self.batch_size,max_seq_len-1))), axis=1)
        batch_labels = self.pad_id*np.ones((self.batch_size,max_seq_len))
        seq_len = np.zeros(self.batch_size)
        
        for i in xrange(self.batch_size):
            with open(os.path.join(self.directory, 'batch' + str(i) + '.txt'), 'r') as f:
                new_sentence = [lines.decode('utf-8').split() for lines in f][self.batch_id]  #niet effecient, moet beter voor train.data
            new_sentence = [self._word_to_id[word] for word in new_sentence]
            seqlen = len(new_sentence)
            seq_len[i] = min(seqlen, max_seq_len)

            batch_data[i,0:min(seqlen, max_seq_len)] = new_sentence[0:min(seqlen, max_seq_len)]
            history_data[i,-max_seq_len+1:] = batch_data[i,0:-1]
            if seqlen >= self.history_size:
                self.history[i] = np.array(new_sentence[-self.history_size:])
            else:
                self.history[i] = np.concatenate((self.history[i,seqlen:],np.array(new_sentence)))
            new_sentence.append(self.bos_id)
            batch_labels[i,0:min(seqlen, max_seq_len)] = new_sentence[1:min(seqlen, max_seq_len)+1]
        self.batch_id += 1 
        return batch_data, history_data, batch_labels, seq_len
        
    def print_next_batch(self, max_seq_len):
        batch_data, history_data, batch_labels, seq_len = self.next_batch(max_seq_len)
        reverse = {v: k for k, v in self._word_to_id.iteritems()}
        reverse[self.pad_id] = 'PAD'
        batch_lst = [reverse[int(word_id)] for sentence in batch_data for word_id in sentence]
        targets_lst = [reverse[int(word_id)] for sentence in batch_labels for word_id in sentence]
        history_lst = [reverse[int(word_id)] for sentence in history_data for word_id in sentence]
        for i in xrange(self.batch_size):
            print('batch ' + str(i))
            print(batch_lst[max_seq_len*i:max_seq_len*(i+1)])
            print(targets_lst[max_seq_len*i:max_seq_len*(i+1)])
            print(history_lst[(self.history_size+max_seq_len-1)*i:(self.history_size+max_seq_len-1)*(i+1)])
            print(seq_len[i])
        
    @property
    def unk_id(self):
        return self._unk_id
        
    @property
    def bos_id(self):
        return self._bos_id
        
    @property
    def eos_id(self):
        return self._eos_id
        
    @property
    def pad_id(self):
        return self._pad_id

    @property
    def word_to_id(self):
        return self._word_to_id
        
    @property
    def epoch_size(self):
        return self._epoch_size
        

#class ds_data__with_history2(object):
#    def __init__(self, max_seq_len, batch_size, history_size, data_path, name):
#        assert max_seq_len > 1
#        
#        #reading data
#        path = os.path.join(data_path, name)
#        with open(path, "r") as f:
#            word_data = f.read().decode("utf-8").split("\n")[:-1]
#        self.amount_sentences = len(word_data)
#        for i in xrange(self.amount_sentences):
#            word_data[i] = word_data[i].split()
#        #reading word_to_id
#        dict_path = os.path.join(data_path, "dictionary.ds.dict")
#        dictionary = corpora.Dictionary.load(dict_path)
#        self._word_to_id = dict()
#        for (wordid,word) in dictionary.iteritems():
#            self._word_to_id[word] = wordid
#        self._unk_id = self._word_to_id['<unk>']
#        self._bos_id = self._word_to_id['<bos>']
#        self._eos_id = self._word_to_id['<eos>']
#        self._pad_id = len(self._word_to_id)
#
#        self.id_data = self.pad_id*np.ones((self.amount_sentences,max_seq_len))
#        self.history = self.pad_id*np.ones((self.amount_sentences,history_size+max_seq_len-1))
#        self.targets = self.pad_id*np.ones((self.amount_sentences,max_seq_len))
#        self.seq_len = np.zeros(self.amount_sentences)
#        for i in xrange(self.amount_sentences):
#            self.id_data[i,0] = self._word_to_id[word_data[i][0]]
#            if len(word_data[i]) <= max_seq_len:
#                for j in xrange(1,len(word_data[i])):
#                    self.id_data[i,j] = self._word_to_id[word_data[i][j]]
#                    self.targets[i,j-1] = self._word_to_id[word_data[i][j]]
#                self.targets[i,len(word_data[i])-1] = self.bos_id
#                self.seq_len[i] = len(word_data[i])
#                self.history[i, -max_seq_len-1:-1] = self.id_data[i]
#                counter = num_history
#                k = i -1
#                while (counter > 0) and (k > -1):
#                    for l in [1]:
#                        pass
#            else:
#                for j in xrange(1,max_seq_len):
#                    self.id_data[i,j] = self._word_to_id[word_data[i][j]]
#                    self.targets[i,j-1] = self._word_to_id[word_data[i][j]]
#                self.targets[i,max_seq_len-1] = self._word_to_id[word_data[i][max_seq_len]]
#                self.seq_len[i] = max_seq_len
#    
#        self._epoch_size = self.amount_sentences//batch_size
#        self.amount_sentences = self.epoch_size*batch_size
#        self.batch_id = 0
#        
#
#    def next_batch(self):
#        if self.batch_id == self.epoch_size:
#            self.batch_id = 0
#        batch_data = self.id_data[[self.batch_id + i for i in xrange(0,self.amount_sentences,self.epoch_size)]] 
#        batch_labels = self.targets[[self.batch_id + i for i in xrange(0,self.amount_sentences,self.epoch_size)]]
#        batch_history = self.history[[self.batch_id + i for i in xrange(0,self.amount_sentences,self.epoch_size)]]                                   
#        batch_seqlen = self.seq_len[[self.batch_id + i for i in xrange(0,self.amount_sentences,self.epoch_size)]]
#        self.batch_id += 1 
#        return batch_data, batch_labels, batch_history, batch_seqlen
#        
#    @property
#    def unk_id(self):
#        return self._unk_id
#        
#    @property
#    def bos_id(self):
#        return self._bos_id
#        
#    @property
#    def eos_id(self):
#        return self._eos_id
#        
#    @property
#    def pad_id(self):
#        return self._pad_id
#
#    @property
#    def word_to_id(self):
#        return self._word_to_id
#        
#    @property
#    def epoch_size(self):
#        return self._epoch_size
   
  
def ds_raw_data(data_path):
    
    train_path = os.path.join(data_path, "ds.train.txt")
    valid_path = os.path.join(data_path, "ds.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    max_length = calc_max_length(train_path,valid_path,test_path)
    
    lda_path = os.path.join(data_path, "lda.ds.model")
    dict_path = os.path.join(data_path, "dictionary.ds.dict")
    
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
