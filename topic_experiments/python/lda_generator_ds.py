##### comments

# first run transform_ds.py before running this file

##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from gensim import corpora, models
import numpy as np
import string
import argparse

##### flags

ap = argparse.ArgumentParser()
ap.add_argument('--nb_topics', default=75, type=int)
ap.add_argument('--sentences_per_document', default=75, type=int)

opts = ap.parse_args()
nb_topics = opts.nb_topics
sentences_per_document = opts.sentences_per_document

##### settings

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(os.path.split(python_path)[0])[0]
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
    train_path = os.path.join(data_path, "ds.train.txt")

    docs = read_and_split_doc(train_path, sentences_per_document)
    texts = [[word for word in doc.split()] for doc in docs]
    dictionary = corpora.dictionary.Dictionary(texts)
    corpus = corpus_iterator(texts,dictionary)

    corpora.MmCorpus.serialize('bow_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds.mm',corpus)
    mm = corpora.MmCorpus('bow_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds.mm')
    tfidf = models.TfidfModel(mm,id2word=dictionary,normalize=True)
    corpora.MmCorpus.serialize('tfidf_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds.mm', tfidf[mm], progress_cnt=10000)

    tfidf.save(tf_save_path)

    mm = corpora.MmCorpus('tfidf_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds.mm')
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=nb_topics)
    lda.save(lda_save_path)
    lda_dict = dictionary
    lda_dict.save(dict_save_path)

    return lda, lda_dict

##### script

lda_save_path = os.path.join(data_path,'topic_experiments', 'lda_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds')
dict_save_path = os.path.join(data_path, 'topic_experiments', 'dictionary_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds')
tf_save_path = os.path.join(data_path, 'topic_experiments', 'tfidf_'+str(nb_topics)+'_'+str(sentences_per_document)+'.ds')
lda, lda_dict = lda_generate_model(sentences_per_document, nb_topics, data_path, lda_save_path, dict_save_path,tf_save_path)

print(str(nb_topics)+ ' topics are generated based on documents of ' + str(sentences_per_document) + ' sentences long')

nb_topics_to_print = 10
nb_words_per_topic_to_print = 20

nb_topics_to_print_tex = 6
nb_words_per_topic_to_print_tex = 20
with open('topics_'+str(nb_topics)+'_'+str(sentences_per_document)+'.tex','w') as f:
    f.write('\\begin{table}[H]\n')
    f.write('\\centering\n')
    f.write('\\caption[Number of topics = '+str(nb_topics)+', sentences per document = '+str(sentences_per_document)+']{Number of topics = '+str(nb_topics)+', sentences per document = '+str(sentences_per_document)+'}'+'\n')
    f.write('\\label{tab:topics_'+str(nb_topics)+'_'+str(sentences_per_document)+'}\n')

    tex_str = '\\begin{tabular}{'
    for j in xrange(nb_topics_to_print_tex):
        tex_str = tex_str + '|c'
    tex_str = tex_str + '|}'
    f.write(tex_str+'\n')

    f.write('\\hline\n')

    tex_str = ''
    for j in xrange(nb_topics_to_print_tex-1):
        tex_str = tex_str + 'Topic ' + str(j+1) + ' & '
    tex_str = tex_str + 'Topic ' + str(nb_topics_to_print_tex) + ' \\\\ \\hline \\hline'
    f.write(tex_str+'\n')

    topic_word_list = []
    for i in xrange(nb_topics_to_print_tex):
        current_topic_word_list = []
        for j in [k for (k,l) in lda.show_topic(i,topn=nb_words_per_topic_to_print_tex)]:
            current_topic_word_list.append(j.encode('utf-8'))
        topic_word_list.append(current_topic_word_list)

    for i in xrange(nb_words_per_topic_to_print_tex):
        tex_str = ''
        for j in xrange(nb_topics_to_print_tex-1):
            tex_str = tex_str + str(topic_word_list[j][i] + ' & ')
        tex_str = tex_str + str(topic_word_list[nb_topics_to_print_tex-1][i]) + '\\\\'
        f.write(tex_str+'\n')

    f.write('\hline\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
        


