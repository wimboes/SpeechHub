##### comments

# first run transform_ds.py before running this file

##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models
import numpy as np
import string

##### settings

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
data_path = os.path.join(general_path,'input')

nb_topics = 100
sentences_per_document = 100

##### functions and classes
  
def read_and_split_doc(path,sentences_per_document):
    with tf.gfile.GFile(path, "r") as f:
        sentences = f.read().decode("ascii", 'ignore').split("\n")
    nb_sentences = len(sentences)
    nb_docs = int(np.floor(nb_sentences/sentences_per_document)) # words per small document
    split_doc = []
    for i in xrange(nb_docs):
        split_doc.append(string.join(sentences[i*sentences_per_document:(i+1)*sentences_per_document],sep=" "))
    if nb_sentences % sentences_per_document:
        split_doc.append(string.join(sentences[nb_docs*sentences_per_document:nb_sentences],sep=" "))
    return split_doc

class corpus_iterator(object):
    def __init__(self,texts,dictionary):
        self.corpus = texts
        self.dict = dictionary
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.dict.doc2bow(self.corpus[i])

def lda_generate_model(sentences_per_document, nb_topics, data_path, lda_save_path, dict_save_path):
    train_path = os.path.join(data_path, "train.txt")

    docs = read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.dictionary.Dictionary(texts)
    corpus = corpus_iterator(texts,dictionary)

    corpora.MmCorpus.serialize('bow.mm',corpus)
    mm = corpora.MmCorpus('bow.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('tfidf.mm', tfidf[mm], progress_cnt=10000)

    mm = corpora.MmCorpus('tfidf.mm')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=nb_topics)
    lda.save(lda_save_path)
    lda_dict = dictionary
    lda_dict.save(dict_save_path)

    return lda, lda_dict

##### script

lda_save_path = os.path.join(data_path, "lda.model")
dict_save_path = os.path.join(data_path, "dictionary.dict")
lda, lda_dict = lda_generate_model(sentences_per_document, nb_topics, data_path, lda_save_path, dict_save_path)