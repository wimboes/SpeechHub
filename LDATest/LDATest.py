from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from gensim import corpora, models

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
         a = f.read().replace("\n", "<eos>")
         #a = a.split()
         return a     
         
document1= read_words('/home/wim/SpeechHub/LDATest/ptb.test.txt')
documents=[document1]
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)

class MyCorpus(object):
    def __init__(self,texts,dictionary):
        self.corpus = texts
        self.dict = dictionary
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.dict.doc2bow(self.corpus[i])

#doc2bowvec = dictionary.doc2bow(texts[0])
corpus = MyCorpus(texts,dictionary)
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)