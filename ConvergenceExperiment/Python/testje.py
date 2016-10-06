from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import tensorflow as tf
import reader

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'Input')
output_path = os.path.join(general_path,'Output')
config_path = os.path.join(general_path,'Configurations')
global_path = os.path.join(os.path.split(general_path)[0],'Global')

sys.path.append(config_path)
sys.path.append(global_path)
from config0 import *

train_word_data, validation_word_data, test_word_data, _ = reader.ptb_raw_data(input_path)
epoch_size = ((len(train_word_data) // batch_size) - 1) // num_unrolls
train_data, train_targets = reader.ptb_producer(train_word_data,batch_size,num_unrolls)
validation_data, validation_targets = reader.ptb_producer(validation_word_data,batch_size,num_unrolls)
test_data, test_targets = reader.ptb_producer(test_word_data,batch_size,num_unrolls)

keep_prob_placeholder = tf.placeholder(tf.float32)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob_placeholder)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)        
initial_state = cell.zero_state(batch_size, data_type)

embedding = tf.Variable(tf.zeros([vocab_size, embedded_size]))
input_placeholder = tf.placeholder(tf.int32)
input_target_placeholder = tf.placeholder(tf.int32)
inputs = tf.nn.embedding_lookup(embedding, input_placeholder)
inputs = tf.nn.dropout(inputs, keep_prob_placeholder)