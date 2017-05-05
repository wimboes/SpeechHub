##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

from gensim import corpora, models
import numpy as np
import collections

##### functions
        
class ds_data_sentence(object):
    def __init__(self, batch_size, dict_path, data_path, name):
        path = os.path.join(data_path, name)
        
        self._sentences = [line.decode('utf-8') for line in open(path)]
        
        amount_sentences = len(self._sentences)
        self._longest_sentence = max(len(line.split()) for line in self._sentences)
        self._epoch_size = amount_sentences//batch_size
        self.batch_size = batch_size
        if batch_size !=1:
            sys.exit("Not possible with this batch size")
            
        dict_path = os.path.join(dict_path, "dictionary.ds")
        dictionary = corpora.Dictionary.load(dict_path)
        self._word_to_id = dict()
        for (wordid,word) in dictionary.iteritems():
            self._word_to_id[word] = wordid
        self._unk_id = self._word_to_id['<UNK>']
        self._bos_id = self._word_to_id['<s>']
        self._eos_id = self._word_to_id['</s>']
        self._pad_id = len(self._word_to_id)
        
        self.batch_id = 0
        

    def next_batch(self, max_seq_len):
        if self.batch_id == self.epoch_size:
            self.batch_id = 0
        batch_data = self.pad_id*np.ones((self.batch_size,max_seq_len))
        batch_labels = self.pad_id*np.ones((self.batch_size,max_seq_len))
        seq_len = np.zeros(self.batch_size)
        
        sentences = [self._sentences[self.batch_id]]      
        i = 0 
        new_sentence = sentences[i].split()
        new_sentence = [self._word_to_id[word] for word in new_sentence]
        seqlen = len(new_sentence)-1
        seq_len[i] = min(seqlen, max_seq_len)
            
        batch_data[i,0:min(seqlen, max_seq_len)] = new_sentence[0:min(seqlen, max_seq_len)]
        new_sentence.append(self.pad_id)
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
            
    def assign_batch_id(self, value):
        self.batch_id = value
        
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
    def longest_sentence(self):
        return self._longest_sentence
        
    @property
    def epoch_size(self):
        return self._epoch_size
        
class ds_data_continuous(object):
    def __init__(self, batch_size, num_steps, dict_path, data_path, name):
        path = os.path.join(data_path, name)
        self.batch_size = batch_size
        self.num_steps = num_steps
        if (batch_size !=1) or (num_steps !=1):
            sys.exit("Not possible with this batch size")        
        
        
        self._words = [lines.decode('utf-8').split()[:-1] for lines in open(path)]
        self._words = [item for sublist in self._words for item in sublist]
        self._epoch_size = len(self._words)
        self._words.append('<s>')
            
        #reading word_to_id
        dict_path = os.path.join(dict_path, "dictionary.ds")
        dictionary = corpora.Dictionary.load(dict_path)
        self._word_to_id = dict()
        for (wordid,word) in dictionary.iteritems():
            self._word_to_id[word] = wordid
        self._unk_id = self._word_to_id['<UNK>']
        self._bos_id = self._word_to_id['<s>']
        self._eos_id = self._word_to_id['</s>']
        self._pad_id = len(self._word_to_id)
        
        self.batch_id = 0
        

    def next_batch(self):
        if self.batch_id == self.epoch_size:
            self.batch_id = 0
        batch_data = np.array([[self.word_to_id[self._words[self.batch_id]]]])
        batch_labels = np.array([map(lambda x:x if x!= self._bos_id else self._eos_id,[self.word_to_id[self._words[self.batch_id+1]]])])
        self.batch_id += 1 
        return batch_data, batch_labels
        
    def print_next_batch(self):
        batch_data, batch_labels = self.next_batch()
        reverse = {v: k for k, v in self._word_to_id.iteritems()}
        reverse[self.pad_id] = 'PAD'
        batch_lst = [reverse[int(word_id)] for sentence in batch_data for word_id in sentence]
        targets_lst = [reverse[int(word_id)] for sentence in batch_labels for word_id in sentence]
        for i in xrange(self.batch_size):
            print('batch ' + str(i))
            print(batch_lst[self.num_steps*i:self.num_steps*(i+1)])
            print(targets_lst[self.num_steps*i:self.num_steps*(i+1)])

    def assign_batch_id(self, value):
        self.batch_id = value
        
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

class ds_data_sentence_with_history(object):
    def __init__(self, batch_size, history_size,dict_path, data_path, name):
        self.history_size = history_size        

        #reading data
        path = os.path.join(data_path, name)
        
        self._sentences = [line.decode('utf-8') for line in open(path)]
        
        amount_sentences = len(self._sentences)
        self._longest_sentence = max(len(line.split()) for line in self._sentences)
        self._epoch_size = amount_sentences//batch_size
        self.batch_size = batch_size
        if batch_size !=1:
            sys.exit("Not possible with this batch size")
            
        #reading word_to_id
        dict_path_ = os.path.join(dict_path, "dictionary.ds")
        dictionary = corpora.Dictionary.load(dict_path_)
        self._word_to_id = dict()
        for (wordid,word) in dictionary.iteritems():
            self._word_to_id[word] = wordid

        self._unk_id = self._word_to_id['<UNK>']
        self._bos_id = self._word_to_id['<s>']
        self._eos_id = self._word_to_id['</s>']
        self._pad_id = len(self._word_to_id)
        
        tfidf_path = os.path.join(dict_path,"tfidf.ds")
        self._tfidf = models.TfidfModel.load(tfidf_path)
        self._idfs = self._tfidf.idfs
        self._idfs[self._pad_id] = 0

        
        self.batch_id = 0
        self.history = self.pad_id*np.ones((batch_size,history_size))
        

    def next_batch(self, max_seq_len):
        if self.batch_id == self.epoch_size:
            self.batch_id = 0
        batch_data = self.pad_id*np.ones((self.batch_size,max_seq_len))
        history_data = np.concatenate((self.history,self.pad_id*np.ones((self.batch_size,max_seq_len-1))), axis=1)
        history_tfidf = np.zeros(np.shape(history_data)) 
        batch_labels = self.pad_id*np.ones((self.batch_size,max_seq_len))
        seq_len = np.zeros(self.batch_size)
        
        sentences = [self._sentences[self.batch_id]]
        i = 0
        new_sentence = sentences[i].split()
        new_sentence = [self._word_to_id[word] for word in new_sentence]
        seqlen = len(new_sentence) - 1
        seq_len[i] = min(seqlen, max_seq_len)
        
        batch_data[i,0:min(seqlen, max_seq_len)] = new_sentence[0:min(seqlen, max_seq_len)]
        history_data[i,-max_seq_len+1:] = batch_data[i,0:-1]   
            
        history_tfidf[i,:] = [self._idfs[id] for id in history_data[i,:]]
        if seqlen >= self.history_size:
            self.history[i] = np.array(new_sentence[-self.history_size-1:-1])
        else:
            self.history[i] = np.concatenate((self.history[i,seqlen:],np.array(new_sentence[:-1])))
        new_sentence.append(self.pad_id)
        batch_labels[i,0:min(seqlen, max_seq_len)] = new_sentence[1:min(seqlen, max_seq_len)+1]
        self.batch_id += 1 
        
        return batch_data, history_data, history_tfidf, batch_labels, seq_len
        
    def print_next_batch(self, max_seq_len):
        batch_data, history_data, history_tfidf, batch_labels, seq_len = self.next_batch(max_seq_len)
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
            print(history_tfidf[i,:])
            print(seq_len[i])
        
    def assign_batch_id(self, value):
        self.batch_id = value
        
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
    def tfidf(self):
        return self._tfidf
    
    @property
    def longest_sentence(self):
        return self._longest_sentence
        
    @property
    def epoch_size(self):
        return self._epoch_size
