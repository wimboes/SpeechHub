# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np

if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
        try:
            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)



import tensorflow as tf
import reader
import fnmatch
import shutil
from gensim import corpora, models


python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

# set data and save path

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","original","test_name")

flags.DEFINE_integer("first_n_best_list", 601327, "first_n_best_list")
flags.DEFINE_integer("amount_n_best_list", 2, "amount_n_best_list")
flags.DEFINE_string("name","n-best1","name")



flags.DEFINE_string("loss_function","full_softmax","loss_function")

flags.DEFINE_string("data_path", input_path, "data_path")
flags.DEFINE_string("save_path", output_path, "save_path")
flags.DEFINE_bool("use_fp16", False, "train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class ds_original_model(object):
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        self._num_steps = num_steps = config['num_steps']
        hidden_size = config['hidden_size']
        vocab_size = input_.pad_id 
        embedded_size = config['embedded_size']
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config['num_layers'], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size + 1, embedded_size], dtype=data_type()) #om pad symbool toe te laten
            inputs = tf.nn.embedding_lookup(embedding, data)
        
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, dtype=data_type(), sequence_length = seq_len)
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        
        self._cost, self._nb_words_in_batch = get_loss_function(output, softmax_w, softmax_b, labels, input_, is_training)
        
        self._final_state = state        
        
    @property
    def input(self):
        return self._input

    @property
    def nb_words_in_batch(self):
        return self._nb_words_in_batch    
        
    @property
    def num_steps(self):
        return self._num_steps

    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def data(self):
        return self._data
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def seq_len(self):
        return self._seq_len

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state


def get_loss_function(output, softmax_w, softmax_b, targets, data, is_training):
    targets = tf.reshape(targets, [-1])
    if is_training:
        mask = tf.not_equal(targets,[data.pad_id])
    else:
        mask = tf.logical_and(tf.not_equal(targets,[data.pad_id]),tf.not_equal(targets,[data.unk_id]))
    mask2 = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    output = tf.gather(output, mask2)
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32))

    if FLAGS.loss_function == "full_softmax":
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
        return tf.reduce_sum(loss), nb_words_in_batch

    if FLAGS.loss_function == 'sampled_softmax':
        if is_training:
            loss = tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1,1]), 32, data.pad_id)
            return tf.reduce_sum(loss), nb_words_in_batch
        else:
            logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
            return tf.reduce_sum(loss), nb_words_in_batch

    return 0

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0):
    start_time = time.time()
    costs = 0.0
    iters = 0
    processed_words = 0
    state = session.run(model.initial_state)

    fetches = {"cost": model.cost,"nb_words_in_batch": model.nb_words_in_batch, "final_state": model.final_state}

    for step in range(model.input.epoch_size):
        batch_data, batch_labels, batch_seq_len = model.input.next_batch(model.num_steps)
        feed_dict = {}
        feed_dict[model.data] = batch_data
        feed_dict[model.labels] = batch_labels
        feed_dict[model.seq_len] = batch_seq_len
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
    
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            nb_words_in_batch = vals["nb_words_in_batch"]
            state = vals["final_state"]
    
            costs += cost
            iters += nb_words_in_batch 
    return np.exp(costs/iters)

def find_n_best_lists(n_best_list_nr, amount_n_best_list):
    n_best_list = ' /users/spraak/jpeleman/lexsub/asr/comp-k/vl/fv'+str(n_best_list_nr)+'/sri/fv'+str(n_best_list_nr)+'_meddest/nbest/(.)+-best.txt'
    
    possible_files = []
    for path, subdirs, files in os.walk('/users/spraak/jpeleman/lexsub/asr/comp-k/vl/'):
	for file in files:	
		if fnmatch.fnmatch(file, 'fv*-best.txt'):
        		possible_files.append(os.path.join(path,file))
    
    examined_files = []
    for i in range(n_best_list_nr, n_best_list_nr + amount_n_best_list):
	for file in possible_files:
	    if fnmatch.fnmatch(file, '*fv' + str(i) + '*'):
	        examined_files.append(file)
    examined_files.sort()
    return examined_files

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
	

def main(_):
    print('N-best list rescoring started')

    dict_path = os.path.join(FLAGS.data_path, "dictionary.ds")
    dictionary = corpora.Dictionary.load(dict_path)
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
            word_to_id[word] = wordid   

    param_np = np.load((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'))
    param_np = param_np['param_train_np']
    
    param =  ['num_layers', 'hidden_size', 'embedded_size']
    
    eval_config = {}
    eval_config['batch_size'] = 1
    for i in range(0,len(param_np)):
        if param_np[i][0] in param:
            eval_config[param_np[i][0]] = int(param_np[i][1])

    n_best_files = find_n_best_lists(FLAGS.first_n_best_list, FLAGS.amount_n_best_list)
    best_sentence_history = []
    for file in n_best_files:
	print(file)
	sentences = []
	#read all sentence hypotheses	
	with open(file,'r') as f:
            for i in range(5): sentences.append(f.readline().decode('utf-8'))	
	
        #make sentence files that later needs to be grades
	accoustic_score = [float(sentence.split()[0]) for sentence in sentences]	
	
	if os.path.exists(os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1])):
	    filelist = [ f for f in os.listdir(os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1]))]
            for f in filelist:
               	remove(os.path.join(os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1]),f))
	else:
	    os.mkdir(os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1]))
	
	for i in range(len(sentences)):
	    sen = sentences[i].split()
	    new_sen = []
            for j in range(len(sen)): 
		if sen[j] in word_to_id.keys():
	   	    new_sen.append(sen[j].encode('utf-8'))
	    new_sen = ' '.join(new_sen)
	    with open(os.path.join(os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1]),'hypothesis'+str(i)+'.txt'),'w') as g:
	        for best_sen in best_sentence_history: g.write(best_sen +'\n')
                g.write(new_sen) 	
	
	#compute perplexity
	ppl = []
        for i in range(len(sentences)):
	    eval_name = 'hypothesis'+str(i)+'.txt'

	    with tf.Graph().as_default():
                with tf.name_scope("test"):
                    eval_data = reader.ds_data_sentence(eval_config['batch_size'], FLAGS.data_path, os.path.join(FLAGS.save_path,os.path.split(file[:-4])[1]), eval_name)
                    eval_config['num_steps'] = eval_data.longest_sentence
                    with tf.variable_scope("model"):
                        mtest = ds_original_model(is_training=False, config=eval_config, input_=eval_data)
			
                sv = tf.train.Supervisor(summary_writer=None, save_model_secs=0, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
                with sv.managed_session() as session:
                    test_perplexity=  run_epoch(session, mtest)
            print("hypothesis %d with PPL %.3f" % (i,test_perplexity))

            ppl.append(test_perplexity)

	vals = np.array(ppl) + np.array(accoustic_score)	
	sort_index = np.argsort(vals)[::-1]


	#best sentence:
	sen = sentences[sort_index[0]].split()
	new_sen = []
        for i in range(len(sen)): 
	    if sen[i] in word_to_id.keys():
	        new_sen.append(sen[i].encode('utf-8'))
	new_sen = ' '.join(new_sen)
	best_sentence_history.append(new_sen)

	sentences2 = []
	for i in sort_index:
	    sen = sentences[i].split()
	    sen[1] = str(ppl[i])
	    sen = ' '.join(sen)
	    sentences2.append(sen.encode('utf-8'))

	with open(os.path.join(FLAGS.save_path,os.path.split(file)[1]),'w') as h:
	    for sentence in sentences2: h.write(sentence +'\n')

    
    print('done')
    
if __name__ == "__main__":
    tf.app.run()
