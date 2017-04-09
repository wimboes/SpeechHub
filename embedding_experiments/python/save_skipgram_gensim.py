import gensim
import os
import numpy as np

input_path = '/esat/spchtemp/scratch/robbe/SpeechHub/input'

model = gensim.models.Word2Vec.load(os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0],'output/embedding_skip_128/embedding_skip.emb'))

dict_path = os.path.join(input_path, "dictionary.ds")
dictionary = gensim.corpora.Dictionary.load(dict_path)

word_to_id = dict()
for (wordid,word) in dictionary.iteritems():
    word_to_id[word] = wordid

id_to_word = {v: k for k, v in word_to_id.iteritems()}

vocab_size = len(word_to_id)

# +1 for padding symbol
embedding = np.zeros([vocab_size+1,np.size(model['<UNK>'])])
for id in range(vocab_size):
    embedding[id,:] = model[id_to_word[id].encode('utf-8')]

np.save('embedding_128.npy',embedding)
