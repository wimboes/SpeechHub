
##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models

##### topic parameters

nb_topics = 100

##### data path

data_path = '/home/wim/SpeechHub/LDA/Input'

##### script

def _read_words_no_split(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>")
        
class MyCorpus(object):
    def __init__(self,texts,dictionary):
        self.corpus = texts
        self.dict = dictionary
    def __iter__(self):
        for i in range(len(self.corpus)):
            yield self.dict.doc2bow(self.corpus[i])
            
def lda_generate_model(data_path=None,lda_save_path='lda.model',dict_save_path='dictionary.dict'):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")

    doctrain = _read_words_no_split(train_path)
    docvalid = _read_words_no_split(valid_path)
    
    docs = [doctrain,docvalid]
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.Dictionary(texts)

    corpus = MyCorpus(texts,dictionary)
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=nb_topics)
    
    lda.save(lda_save_path)
    dictionary.save(dict_save_path)
    
    return
    
lda_generate_model(data_path=data_path)
    
