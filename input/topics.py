from gensim import models, corpora

lda = models.LdaModel.load('lda_512_10.ds')
dic = corpora.Dictionary.load('dictionary.ds')

#word_to_id = dict()
#for (wordid,word) in dic.iteritems():
#	word_to_id[word] = wordid

nb_topics = lda.num_topics

for i in range(nb_topics):
	lst = lda.get_topic_terms(i,topn=10)
	lal = [dic[tup[0]] for tup in lst]
	print('topic ' + str(i))
	print(lal)
