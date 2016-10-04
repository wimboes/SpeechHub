# authors : Wim Boes & Robbe Van Rompaey
# date: 4-10-2016 11:00

# imports

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

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.concat(1, outputs), [-1, size])
		softmax_w = tf.get_variable(
				"softmax_w", [size, vocab_size], dtype=data_type())
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(input_.targets, [-1])],
				[tf.ones([batch_size * num_steps], dtype=data_type())])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
																			config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(
				tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op


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
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to PTB data directory")

	raw_data = reader.ptb_raw_data(FLAGS.data_path)
	train_data, valid_data, test_data, _ = raw_data

	config = get_config()
	eval_config = get_config()
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

def main():
	
	return

	
class Model:
	
	def __init__(self,config, train_word_data, validation_word_data, test_word_data):
		self.init_scale = config['init_scale']
		self.learning_rate = config['learning_rate']
		self.max_grad_norm = config['max_grad_norm']
		self.num_layers = config['num_layers']
		self.num_unrolls = config['num_unrolls']
		self.hidden_size = config['hidden_size']
		self.init_epochs = config['init_epochs']
		self.max_epochs = config['max_epochs']
		self.keep_prob = config['keep_prob']
		self.lr_decay = config['lr_decay']
		self.batch_size = config['batch_size']
		self.vocab_size = config['vocab_size']
		self.embedded_size = config['embedded_size']
		self.data_type = config['data_type']
		self.epoch_size = ((len(train_word_data) // self.batch_size) - 1) // self.num_unrolls

		self.train_data, self.train_targets = reader.ptb_producer(train_word_data,self.batch_size,self.num_unrolls)
		self.validation_data, self.validation_targets = reader.ptb_producer(validation_word_data,self.batch_size,self.num_unrolls)
		self.test_data, self.test_targets = reader.ptb_producer(test_word_data,self.batch_size,self.num_unrolls)
		
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True)
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
		self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)		
		self.state = cell.zero_state(batch_size, self.data_type)
		
		self.embedding = tf.Variable(tf.zeros([self.vocab_size, self.hidden_size]),name="embedding")
		self.softmax_w = tf.Variable(tf.zeros([self.hidden_size, self.vocab_size]),name="softmax_w")
		self.softmax_b = tf.Variable(tf.zeros([self.vocab_size]),name="softmax_b")
		
		train_inputs = tf.nn.embedding_lookup(self.embedding, self.train_data)
		validation_inputs = tf.nn.embedding_lookup(self.embedding, self.validation_data)
		test_inputs = tf.nn.embedding_lookup(self.embedding, self.test_data)
		
		logits = tf.matmul(output, softmax_w) + softmax_b
		self.loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
		
# 		train_inputs = tf.nn.embedding_lookup(embedding, train_data)
# 		validation_inputs = tf.nn.embedding_lookup(embedding, validation_data)
# 		test_inputs = tf.nn.embedding_lookup(embedding, test_data)

# 		inputs = tf.nn.dropout(inputs, config.keep_prob)

if __name__ == "__main__":
	train_word_data, valid_word_data, test_word_data, _ = reader.ptb_raw_data(input_path)
	model = Model(config, train_word_data, validation_word_data, test_word_data)
	with tf.Session() as sess:
		Model.trainvanzelf()