##### imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf


##### settings

data_path = '/home/robbe/SpeechHub/Ptbsentence'

nb_vocab = 10000

##### functions

def create_all_files(data_path,nb_vocab):
        train_path = os.path.join(data_path, "Input/ptb.train.txt")
        
        word_to_id = build_vocab(train_path,nb_vocab)
        
        create_file(os.path.join(data_path, "Input/ptb.train.txt"), word_to_id, os.path.join(data_path, 'train.txt'))
        create_file(os.path.join(data_path, "Input/ptb.valid.txt"), word_to_id, os.path.join(data_path, 'valid.txt'))
        create_file(os.path.join(data_path, "Input/ptb.test.txt"), word_to_id, os.path.join(data_path, 'test.txt'))


def create_file(path, word_to_id, save_path):
    sentences = read_sentences(path) 
    f = open(save_path, "w")
    word_to_id_set = set(word_to_id.keys())
    for i in xrange(len(sentences)):
        words =  sentences[i].lower().split()
        for j in range(len(words)):
            if words[j] in word_to_id_set:
                f.write(words[j])
            else:
                f.write('<unk>')
            if j == len(words)-1 and i != len(sentences)-1:
                f.write('\n')
            elif j != len(words)-1:
                f.write(' ')
    f.close()
        
def read_sentences(path):
	with tf.gfile.GFile(path, "r") as f:
		return f.read().decode("ascii", 'ignore').replace("\n", "<eos> \n <bos> ").split("\n")[0:-1]

def build_vocab(path, nb_vocab):
    sentences = read_sentences(path)
    words = [word for sentence in sentences for word in sentence.lower().split()]

    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    words = words[0:nb_vocab-1]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

##### script
    
create_all_files(data_path,nb_vocab)

