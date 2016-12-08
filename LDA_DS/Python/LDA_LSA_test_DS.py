
##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from gensim import corpora, models

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

    corpora.MmCorpus.serialize('ptb_bow.mm',corpus)
    mm = corpora.MmCorpus('ptb_bow.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('ptb_tfidf.mm', tfidf[mm], progress_cnt=10000)

    mm = corpora.MmCorpus('ptb_tfidf.mm')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=10)
    lda_dict = dictionary
    
    return lda, lda_dict
    
def lsa_generate_model(data_path=None,lda_save_path='lda.model',dict_save_path='dictionary.dict'):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")

    doctrain = _read_words_no_split(train_path)
    docvalid = _read_words_no_split(valid_path)
    
    docs = [doctrain,docvalid]
    texts = [[word for word in doc.lower().split()] for doc in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = MyCorpus(texts,dictionary)

    corpora.MmCorpus.serialize('ptb_bow.mm',corpus)
    mm = corpora.MmCorpus('ptb_bow.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('ptb_tfidf.mm', tfidf[mm], progress_cnt=10000)
    
    mm = corpora.MmCorpus('ptb_tfidf.mm')
    lsa = models.lsimodel.LsiModel(corpus=mm, num_topics=5,id2word=dictionary)
    lsa_dict = dictionary
    
    return lsa, lsa_dict
    
lda, lda_dict = lda_generate_model(data_path=data_path)
lda_txt = open("lda.txt", "w")
for i in xrange(10):
    lda_txt.write("topic %d\n" % i)
    for j in xrange(25):
        lda_txt.write("%s : %f\n" % (lda_dict[lda.get_topic_terms(i,topn=25)[j][0]],lda.get_topic_terms(i,topn=25)[j][1]))
    lda_txt.write("\n")
lda_txt.close()


lsa, lsa_dict = lsa_generate_model(data_path=data_path)   
lsa_txt = open("lsa.txt", "w")
for i in xrange(10):
    lsa_txt.write("topic %d\n" % i)
    for j in xrange(25):
        lsa_txt.write("%s : %f\n" % (lsa.show_topic(i,topn=25)[j][0],lsa.show_topic(i,topn=25)[j][1]))
    lsa_txt.write("\n")
lda_txt.close()