### authors : Wim Boes & Robbe Van Rompaey
### date: 4-10-2016 

### imports 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import tensorflow as tf
import reader

# loading and batching train, validation and test data
train_word_data, validation_word_data, test_word_data, _ = reader.ptb_raw_data(input_path)
epoch_size = ((len(train_word_data) // batch_size) - 1) // num_unrolls
train_data, train_targets = reader.ptb_producer(train_word_data,batch_size,num_unrolls)
validation_data, validation_targets = reader.ptb_producer(validation_word_data,batch_size,num_unrolls)
test_data, test_targets = reader.ptb_producer(test_word_data,batch_size,num_unrolls)

# making lstm cell and initializing state to 0
keep_prob_placeholder = tf.placeholder(tf.float32)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob_placeholder)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)		
initial_state = cell.zero_state(batch_size, data_type)

# embeddings
embedding = tf.Variable(tf.zeros([vocab_size, embedded_size]))
# input_placeholder = tf.placeholder(tf.int32,shape=(batch_size,num_unrolls))
# input_target_placeholder = tf.placeholder(tf.int32,shape=(batch_size,num_unrolls))
inputs = tf.nn.embedding_lookup(embedding, train_data)
inputs = tf.nn.dropout(inputs, keep_prob_placeholder)

# output model + loss
outputs = []
state = initial_state
for num_unroll in xrange(num_unrolls):
	if num_unroll > 0: 
		tf.get_variable_scope().reuse_variables()
#	if num_unroll == 0:
#		(cell_output, state) = cell(inputs[:, num_unroll, :], initial_state)
	(cell_output, state) = cell(inputs[:, num_unroll, :], state)
	outputs.append(cell_output)
final_state = state
	
# size?
output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
softmax_w = tf.Variable(tf.zeros([hidden_size, vocab_size]))
softmax_b = tf.Variable(tf.zeros([vocab_size]))
logits = tf.matmul(output, softmax_w) + softmax_b
loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(train_targets, [-1])],
          [tf.ones([batch_size * num_unrolls], dtype=data_type)])
cost = tf.reduce_sum(loss) / batch_size

# optimizer
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(1e-8)
# global step?
train_op = optimizer.apply_gradients(zip(grads, tvars))
lr_placeholder = tf.placeholder(tf.float32)
lr_update = tf.assign(lr,lr_placeholder)

# session
initializer = tf.random_uniform_initializer(-init_scale, init_scale)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	# training
	for epoch in xrange(max_epochs):
 		state = sess.run(initial_state)
		lr_decay = lr_decay ** max(epoch - init_epochs, 0.0)
		learning_rate = learning_rate * lr_decay
		print("Epoch %d:" % epoch)
		for batch in xrange(epoch_size):
			feed_dict = {lr_placeholder : learning_rate,
						 keep_prob_placeholder : keep_prob}
			for i, (c, h) in enumerate(initial_state):
				feed_dict[c] = state[i].c
				feed_dict[h] = state[i].h
			sess.run(train_op,feed_dict)
			print("Batch %d: loss function = %.3f" % (batch,sess.run(loss)))