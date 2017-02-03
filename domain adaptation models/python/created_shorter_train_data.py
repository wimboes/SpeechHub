#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 08:42:14 2016

@author: robbe
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import copy

def read_sentences(filename):
    with tf.gfile.GFile(filename, "r") as f:
         sentences = f.read().decode("ascii", 'ignore').split("\n")
         print(len(sentences))
         return sentences
         
def create_file(path, save_path, max_amount_of_sentences):
    sentences = read_sentences(path) 
    f = open(save_path, "w")
    for i in xrange(max_amount_of_sentences):
        words =  sentences[i].lower().split()
        for j in range(len(words)):
            f.write(words[j])
            if j == len(words)-1 and i != max_amount_of_sentences-1:
                f.write('\n')
            elif j != len(words)-1:
                f.write(' ')
    f.close()

filename = '/home/robbe/SpeechHub/TEST_TUSSENTIJDSE_PRESENTATIE/input/ds.train.txt'
save_path = '/home/robbe/SpeechHub/TEST_TUSSENTIJDSE_PRESENTATIE/input/train_small.txt'

read_sentences(filename)
create_file(filename,save_path, 600000)
read_sentences(save_path)
