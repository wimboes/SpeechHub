##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
        try:
            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)
	
import tensorflow as tf
from codecs import open

##### settings

data_path = '/esat/spchtemp/scratch/wboes/ds_stream_upper/input'

nb_vocab = 50000

##### functions

def create_all_files(data_path,nb_vocab):
        train_path = os.path.join(data_path, "ds.original.train.txt")
        
        word_to_id = build_vocab(train_path,nb_vocab)
        
        create_file(os.path.join(data_path, "ds.original.train.txt"), word_to_id, os.path.join(data_path, 'ds.train.txt'))
        create_file(os.path.join(data_path, "ds.original.valid.txt"), word_to_id, os.path.join(data_path, 'ds.valid.txt'))
        create_file(os.path.join(data_path, "ds.original.test.txt"), word_to_id, os.path.join(data_path, 'ds.test.txt'))


def create_file(path, word_to_id, save_path):
    sentences = read_sentences(path) 
    f = open(save_path, "w", encoding='utf-8')
    word_to_id_set = set(word_to_id.keys())
    for i in xrange(len(sentences)):
        words =  sentences[i].split()
        for j in range(len(words)):
            if words[j] in word_to_id_set:
                f.write(words[j])
            else:
                f.write('<UNK>')
            if j == len(words)-1 and i != len(sentences)-1:
                f.write('\n')
            elif j != len(words)-1:
                f.write(' ')
    f.close()
        
def read_sentences(path):
	with tf.gfile.GFile(path, "r") as f:
		return f.read().decode("utf-8").split("\n")[0:-1]

def build_vocab(path, nb_vocab):
    sentences = read_sentences(path)
    words = [word for sentence in sentences for word in sentence.split()]

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = words[0:nb_vocab-1]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

##### script
    
create_all_files(data_path,nb_vocab)

