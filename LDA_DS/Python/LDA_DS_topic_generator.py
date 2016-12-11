
##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np
import string

##### topic parameters

nb_topics = 100
sentences_per_document = 100
nb_vocab = 25

##### data path

data_path = '/home/robbe/SpeechHub/LDA_DS/Input'

##### script

#def read_words_comma(filename):
#    with tf.gfile.GFile(filename, "r") as f:
#        return f.read().decode("utf-8").replace("<s>", "").replace("</s>","").replace("\n", "").split()
#        
#def read_and_split_doc(filename,split_nb):
#    full_doc = read_words_comma(filename)
#    len_full_doc = len(full_doc)
#    wpsd = int(np.floor(len_full_doc/split_nb))# words per small document
#    split_doc = []
#    for i in xrange(split_nb):
#        split_doc.append(string.join(full_doc[i*wpsd:(i+1)*wpsd],sep=" "))         
#    return split_doc

def read_words_split(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("ascii", 'ignore').replace("<s>", "").replace("</s>","").replace("\n", "").split()
        
def read_and_split_doc(filename,sentences_per_document):
    with tf.gfile.GFile(filename, "r") as f:
        full_doc = f.read().decode("ascii", 'ignore').replace("<s>", "").replace("</s>", "<eos>").split("\n")  
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
    train_path = os.path.join(data_path, "ds.test.txt")
    
    docs = read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.dictionary.Dictionary(texts, prune_at=nb_vocab)
#    dic_size = len(dictionary.keys())
#    dictionary.filter_n_most_frequent(dic_size-nb_vocab)
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
    train_path = os.path.join(data_path, "ds.test.txt")

    docs = read_and_split_doc(train_path, sentences_per_document)
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