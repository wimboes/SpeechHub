from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gensim

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
         a = f.read().decode("utf-8").replace("\n", "<eos>")
         a = a.split()
         return a     
         
corpus= read_words('/home/wim/SpeechHub/LDATest/ptb.test.txt')
dictionary = gensim.corpora.Dictionary(corpus)
print(dictionary)