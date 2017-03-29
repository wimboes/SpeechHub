# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:56:11 2017

@author: robbe
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

from gensim import corpora, models
import numpy as np
import sys
import argparse

##### flags

ap = argparse.ArgumentParser()
ap.add_argument('--model1', default='original', type=str)
ap.add_argument('--model1_num_run', default='0', type=str)
ap.add_argument('--model2', default='original', type=str)
ap.add_argument('--model2_num_run', default='0', type=str)

opts = ap.parse_args()
##### settings

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
output_path = os.path.join(general_path,'output')

output_path1 = os.path.join(output_path,opts.model1 + '_' + opts.model1_num_run + '/eval.txt')
output_path2 = os.path.join(output_path,opts.model2 + '_' + opts.model1_num_run + '/eval.txt')

def read_eval_file(output_path):
    words = []
    targets = []
    probs = [] 
    with open(output_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].split()
        words.append(line[0].decode('utf-8'))
        targets.append(line[2].decode('utf-8'))
        probs.append(float(line[4]))        
    
    return words, targets, probs
    
words1, targets1, probs1 = read_eval_file(output_path1)
words2, targets2, probs2 = read_eval_file(output_path2)

if words1 != words2 or targets1 != targets2:
    sys.exit('Wrong eval-files')
    
with open(os.path.join(output_path, 'figures/comparation_' + opts.model1 + opts.model1_num_run + '_' + opts.model2 + opts.model2_num_run), 'w') as f:
    f.write("{:<15}".format('input'))
    f.write(" | ")
    f.write("{:<15}".format('target'))
    f.write(" | ")
    f.write("{:<15}".format(opts.model1 + opts.model1_num_run))
    f.write(" <-> ")
    f.write("{:<15}".format(opts.model2 + opts.model2_num_run))
    f.write(' || ')
    f.write('best model: 0:first, 1:second')
    f.write("\n")
    best = []
    for i in range(len(words1)):
        f.write("{:<15}".format(words1[i].encode('utf-8')))
        f.write(" | ")
        f.write("{:<15}".format(targets1[i].encode('utf-8')))
        f.write(" | ")
        f.write("{:<15}".format(probs1[i]))
        f.write(" <-> ")
        f.write("{:<15}".format(probs2[i]))
        f.write(" || ")
        best.append(0 if probs1[i]< probs2[i] else 1)
        f.write(str(best[i]))
        f.write("\n")
    f.write('\n')
    f.write('accuracy: ')
    f.write('\n')
    f.write('model 1: ' + opts.model1 + opts.model1_num_run + ' = ')
    acc = sum(best)/float(len(best))
    f.write(str(1- acc)) 
    f.write('\n')
    f.write('model 2: ' + opts.model2 + opts.model2_num_run + ' = ')
    f.write(str(acc))
    f.write('\n')
           
        

