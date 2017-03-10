##### comments

# first run transform_ds.py before running this file

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
import string

##### flags

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("nb_topics", 75, "nb_topics")
flags.DEFINE_integer("sentences_per_document", 3, "sentences_per_document")

FLAGS = flags.FLAGS

##### settings

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
data_path = os.path.join(general_path,'input')

##### functions and classes
  
def read_and_split_doc(path,sentences_per_document):
    with open(path, "r") as f:
        sentences = f.read().decode("utf-8").split("\n")
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

def lda_generate_model(sentences_per_document, nb_topics, data_path, lda_save_path, dict_save_path, tf_save_path):
    train_path = os.path.join(data_path, "ds.test.txt")

    docs = read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.dictionary.Dictionary(texts)
    corpus = corpus_iterator(texts,dictionary)

    corpora.MmCorpus.serialize('bow.ds.mm',corpus)
    mm = corpora.MmCorpus('bow.ds.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('tfidf.ds.mm', tfidf[mm], progress_cnt=10000)

    tfidf_dict = {dictionary.get(id): value for doc in tfidf[corpus] for id, value in doc}
    np.save(tf_save_path, tfidf_dict)


    mm = corpora.MmCorpus('tfidf.ds.mm')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=nb_topics)
    lda.save(lda_save_path)
    lda_dict = dictionary
    lda_dict.save(dict_save_path)

    return lda, lda_dict

##### script

def main(_):
    
    nb_topics = FLAGS.nb_topics
    sentences_per_document = FLAGS.sentences_per_document
    
    lda_save_path = os.path.join(data_path, "lda.ds.model")
    dict_save_path = os.path.join(data_path, "dictionary.ds.dict")
    tf_save_path = os.path.join(data_path, "tfidf.ds.npy")
    lda, lda_dict = lda_generate_model(sentences_per_document, nb_topics, data_path, lda_save_path, dict_save_path,tf_save_path)
    
    print(str(nb_topics)+ ' topics are generated based on documents of ' + str(sentences_per_document) + ' sentences long')
    
    nb_topics_to_print = 10
    nb_words_per_topic_to_print = 20
 
    for i in xrange(nb_topics_to_print):
        print('Topic number %d:' % i)
        word_list = lda.show_topic(i,topn=nb_words_per_topic_to_print)
        for j in [k for (k,l) in word_list]:
            print(j)
        print('')
            
    
if __name__ == "__main__":
    tf.app.run()


