#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 09:15:30 2016

@author: robbe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np
import string


python_path = os.path.abspath(os.getcwd())
data_path = os.path.split(python_path)[0]

nb_topics = 100
sentences_per_document = 100


def _create_data(data_path,nb_vocab):
    input_path = os.path.join(data_path, "Input")
    dirs = os.listdir(input_path)
    
    if ('ds.valid' + str(nb_vocab) + '.txt') in dirs:
        print('files already created')
    else:
        print('files with vocabulary of ' + str(nb_vocab) + ' will be created')
        
        original_train_txt = os.path.join(data_path, "DS_data/ds.train.txt")
        word_to_id = _build_vocab(original_train_txt,nb_vocab)
        
        _create_file(os.path.join(data_path, "DS_data/ds.train.txt"), word_to_id, os.path.join(input_path, 'ds.train' + str(nb_vocab) + '.txt'))
        _create_file(os.path.join(data_path, "DS_data/ds.valid.txt"), word_to_id, os.path.join(input_path, 'ds.valid' + str(nb_vocab) + '.txt'))
        _create_file(os.path.join(data_path, "DS_data/ds.test.txt"), word_to_id, os.path.join(input_path, 'ds.test' + str(nb_vocab) + '.txt'))


def _create_file(original_txt_location,word_to_id, file_txt_location):
    text_data = _read_words(original_txt_location) 
    file_ = open(file_txt_location, "w")
    for i in range(len(text_data)):
        data =  text_data[i].lower().split()
        for j in range(len(data)):
            if data[j] in word_to_id.keys():
                file_.write(data[j])
                file_.write(' ')
            else:
                file_.write('<unk>')
                file_.write(' ')
        file_.write('\n ')
    file_.close()
        
def _read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().decode("ascii", 'ignore').replace("<s>", "").replace("</s>", "").split("\n") 

def _build_vocab(filename, nb_vocab):
    data = _read_words(filename)
    data = [item for sentences in data for item in sentences.lower().split()]

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = words[0:nb_vocab-2]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id
        
def _read_and_split_doc(filename,sentences_per_document):
    full_doc = _read_words(filename) 
    sentences_full_doc = len(full_doc)
    nb_doc = int(np.floor(sentences_full_doc/sentences_per_document)) # words per small document
    split_doc = []
    for i in xrange(nb_doc):
        split_doc.append(string.join(full_doc[i*sentences_per_document:(i+1)*sentences_per_document],sep=""))
    split_doc.append(string.join(full_doc[nb_doc*sentences_per_document:sentences_full_doc],sep=""))
    return split_doc

class MyCorpus(object):
    def __init__(self,texts,dictionary):
        self.corpus = texts
        self.dict = dictionary
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.dict.doc2bow(self.corpus[i])
            
def lda_generate_model(sentences_per_document, nb_topics, nb_vocab, data_path=None,lda_save_path='lda.model',dict_save_path='dictionary.dict'):
    _create_data(data_path,nb_vocab)
    train_path = os.path.join(data_path, "Input/ds.train"  + str(nb_vocab) + ".txt")
    
    docs = _read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.dictionary.Dictionary(texts)
    corpus = MyCorpus(texts,dictionary)

    corpora.MmCorpus.serialize('ds_bow.mm',corpus)
    mm = corpora.MmCorpus('ds_bow.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('ds_tfidf.mm', tfidf[mm], progress_cnt=10000)

    mm = corpora.MmCorpus('ds_tfidf.mm')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=nb_topics)
    lda_dict = dictionary
    
    return lda, lda_dict
    
def lsa_generate_model(sentences_per_document, nb_topics, nb_vocab, data_path=None,lda_save_path='lda.model',dict_save_path='dictionary.dict'):
    _create_data(data_path,nb_vocab)
    train_path = os.path.join(data_path, "Input/ds.train"  + str(nb_vocab) + ".txt")

    docs = _read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.Dictionary(texts, prune_at=nb_vocab)
    corpus = MyCorpus(texts,dictionary)

    corpora.MmCorpus.serialize('ds_bow.mm',corpus)
    mm = corpora.MmCorpus('ds_bow.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('ds_tfidf.mm', tfidf[mm], progress_cnt=10000)
    
    mm = corpora.MmCorpus('ds_tfidf.mm')
    lsa = models.lsimodel.LsiModel(corpus=mm, num_topics=nb_topics,id2word=dictionary)
    lsa_dict = dictionary
    
    return lsa, lsa_dict
    

_create_data(data_path,1000)
_create_data(data_path,10000)
_create_data(data_path,50000)

nb_vocab = 50000

lda, lda_dict = lda_generate_model(sentences_per_document, nb_topics, nb_vocab, data_path=data_path)
lda_txt = open("lda.txt", "w")
for i in xrange(10):
    lda_txt.write("topic %d\n" % i)
    for j in xrange(25):
        lda_txt.write("%s : %f\n" % (lda_dict[lda.get_topic_terms(i,topn=25)[j][0]],lda.get_topic_terms(i,topn=25)[j][1]))
    lda_txt.write("\n")
lda_txt.close()


lsa, lsa_dict = lsa_generate_model(sentences_per_document, nb_topics, nb_vocab, data_path=data_path)   
lsa_txt = open("lsa.txt", "w")
for i in xrange(10):
    lsa_txt.write("topic %d\n" % i)
    for j in xrange(25):
        lsa_txt.write("%s : %f\n" % (lsa.show_topic(i,topn=25)[j][0],lsa.show_topic(i,topn=25)[j][1]))
    lsa_txt.write("\n")
lda_txt.close()
