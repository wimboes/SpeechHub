###

import os
from gensim import corpora

###

dictionary = corpora.Dictionary.load('/esat/spchtemp/scratch/wboes/ds_stream_upper/input/dictionary.ds.dict')
vocabulary = dictionary.values()

with open('/esat/spchtemp/scratch/wboes/ds_stream_upper/input/vocab.txt','w') as f:
	for word in vocabulary:
		f.write(word.encode('utf-8')+'\n')

