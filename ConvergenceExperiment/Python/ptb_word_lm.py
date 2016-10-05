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
embedding = tf.Variable(tf.zeros([vocab_size, hidden_size]))
input_placeholder = tf.placeholder(tf.int32)
input_target_placeholder = tf.placeholder(tf.int32)
inputs = tf.nn.embedding_lookup(embedding, input_placeholder)
inputs = tf.nn.dropout(inputs, keep_prob_placeholder)

# output model + loss
outputs = []
state = initial_state
for num_unroll in range(num_unrolls):
#	if num_unroll > 0: 
#		tf.get_variable_scope().reuse_variables()
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
loss = tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_target_placeholder, [-1])],
          [tf.ones([batch_size * num_unroll], dtype=data_type])
cost = tf.reduce_sum(loss) / batch_size

# optimizer
lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(lr)
# global step?
train_op = optimizer.apply_gradients(zip(grads, tvars),	global_step=tf.contrib.framework.get_or_create_global_step())
lr_placeholder = tf.placeholder(tf.float32)
lr_update = tf.assign(lr,lr_placeholder)
 
########################## VAN HIER VERDER
 
def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)
 
	fetches = {
			"cost": model.cost,
			"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op
 
	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h
 
		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]
 
		costs += cost
		iters += model.input.num_steps
 
		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
						(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 iters * model.input.batch_size / (time.time() - start_time)))
 
	return np.exp(costs / iters)
 
 
 
 
def x(_):

	eval_config.batch_size = 1
	eval_config.num_steps = 1
 
	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,
																								config.init_scale)
 
		with tf.name_scope("Train"):
			train_input = PTBInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				m = PTBModel(is_training=True, config=config, input_=train_input)
			tf.scalar_summary("Training Loss", m.cost)
			tf.scalar_summary("Learning Rate", m.lr)
 
		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
			tf.scalar_summary("Validation Loss", mvalid.cost)
 
		with tf.name_scope("Test"):
			test_input = PTBInput(config=config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = PTBModel(is_training=False, config=eval_config,
												 input_=test_input)
 
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		with sv.managed_session() as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)
 
				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op,
																		 verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
 
			test_perplexity = run_epoch(session, mtest)
			print("Test Perplexity: %.3f" % test_perplexity)
 
			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)