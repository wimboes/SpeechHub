# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:50:45 2017

@author: robbe
"""
import os
import tensorflow as tf
from gensim import corpora

### inputs

testwoorden = ['mama', 'Jezus', 'meisje', 'auto', 'voetbal']

neigbourhoud = 10

test_name = 'embedding'
num_run = '0'


#file searching

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'output')
path = os.path.join(input_path,'_'.join((test_name,num_run)))
dirs = os.listdir(path)

for file_ in dirs:
    tmp = file_.split('.')
    tmp = tmp[0].split('_')
    if len(tmp) == 3:
        vocab_size = int(tmp[1])
        embedded_size = int(tmp[2])
        
data = os.path.join(path,'embeddings_' + str(vocab_size) + '_' + str(embedded_size) + '.ckpt')

###model

embedded_size = 128

batch = tf.placeholder(tf.int32, [None])


embedding = tf.get_variable("embedding", [vocab_size, embedded_size])
normed_embedding = tf.nn.l2_normalize(embedding, dim=1)

batch_array =  tf.nn.embedding_lookup(embedding, batch)
print()
print(batch_array.get_shape())
normed_array = tf.nn.l2_normalize(batch_array, dim=1)

cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))
_, closest_ids = tf.nn.top_k(cosine_similarity, k=neigbourhoud, sorted=True)



dict_path = os.path.join(os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input'), "dictionary.ds")
dictionary = corpora.Dictionary.load(dict_path)
word_to_id = dict()
for (wordid,word) in dictionary.iteritems():
    word_to_id[word] = wordid

saver = tf.train.Saver()



### Session

word_id = [word_to_id[word] for word in testwoorden]

with tf.Session() as sess:
    saver.restore(sess,data)
    closest_ids = sess.run(closest_ids, {batch : word_id})

print([dictionary[id] for id_lst in closest_ids for id in id_lst])
closest_words = [[dictionary[closest_ids[i,j]].encode('utf-8') for j in range(len(closest_ids[0]))] for i in range(len(closest_ids))]
#[dictionary[id].encode('utf-8') for id_lst in closest_ids for id in id_lst]
    
nb_words_to_print = len(testwoorden)
with open('embedding_experiments.tex','w') as f:
    f.write('\\begin{table}[H]\n')
    f.write('\\centering\n')
    f.write('\\caption[Vocabulary size = '+str(vocab_size-1)+', embedding dimension = '+str(embedded_size)+']{Vocabulary size = '+str(vocab_size-1)+', embedding dimension = '+str(embedded_size)+'}'+'\n')
    f.write('\\label{tab:emb_experiments}\n')

    tex_str = '\\begin{tabular}{'
    for j in xrange(nb_words_to_print):
        tex_str = tex_str + '|c'
    tex_str = tex_str + '|}'
    f.write(tex_str+'\n')

    f.write('\\hline\n')

    tex_str = ''
    for j in xrange(nb_words_to_print-1):
        tex_str = tex_str + testwoorden[j] + ' & '
    tex_str = tex_str + testwoorden[nb_words_to_print-1] + ' \\\\ \\hline \\hline'
    f.write(tex_str+'\n')


    for i in xrange(neigbourhoud):
        tex_str = ''
        for j in xrange(nb_words_to_print-1):
            tex_str = tex_str + str(closest_words[j][i] + ' & ')
        tex_str = tex_str + str(closest_words[nb_words_to_print-1][i]) + '\\\\'
        f.write(tex_str+'\n')

    f.write('\hline\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
