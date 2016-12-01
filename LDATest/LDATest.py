from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gensim
from pprint import pprint

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
         a = f.read().replace("\n", "<eos>")
         a = a.split()
         return a     
         
text= read_words('/home/robbe/SpeechHub/LDATest/ptb.test.txt')
corpus=[text]
dictionary = gensim.corpora.Dictionary(corpus)
print(dictionary)
print(dictionary.token2id)
new_vec = dictionary.doc2bow(text)

#lda = gensim.models.LdaModel(corpus, num_topics=10)