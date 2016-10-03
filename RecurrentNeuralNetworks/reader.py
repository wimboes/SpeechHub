# authors: Wim Boes & Robbe Van Rompaey
# date: 3-10-2016 11:08

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import tensorflow as tf


def read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocabulary(filename):
	data = read_words(filename)
	counter = collections.Counter(data)
	countpairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*countpairs))
	wordtoid = dict(zip(words, range(len(words))))
	return wordtoid


def file_to_word_ids(filename, wordtoid):
	data = read_words(filename)
	return [wordtoid[word] for word in data if word in wordtoid]


def ptb_raw_data(datapath=None):
	trainpath = os.path.join(datapath, "ptb.train.txt")
	validpath = os.path.join(datapath, "ptb.valid.txt")
	testpath = os.path.join(datapath, "ptb.test.txt")
	wordtoid = build_vocabulary(trainpath)
	traindata = file_to_word_ids(trainpath, wordtoid)
	validdata = file_to_word_ids(validpath, wordtoid)
	testdata = file_to_word_ids(testpath, wordtoid)
	vocabulary = len(wordtoid)
	return traindata, validdata, testdata, vocabulary


def ptb_producer(rawdata, batchsize, numsteps, name=None):
	with tf.name_scope(name):
		rawdata = tf.convert_to_tensor(rawdata, name="rawdata", dtype=tf.int32)
		datalen = tf.size(rawdata)
		batchlen = datalen // batchsize
		data = tf.reshape(rawdata[0 : batchsize * batchlen],
											[batchsize, batchlen])

		epochsize = (batchlen - 1) // numsteps
		assertion = tf.assert_positive(
				epoch_size,
				message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
		y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
		return x, y
        
        