##### imports
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
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
import reader
from gensim import corpora, models

import matplotlib.pyplot as plt


##### paths

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

##### flags

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","extended_topic_1","test_name")
flags.DEFINE_string("eval_name",'ds.testshort.txt',"eval_name")

flags.DEFINE_string("loss_function","full_softmax","loss_function")

flags.DEFINE_string("data_path",input_path,"data_path")
flags.DEFINE_string("save_path",output_path,"save_path")
flags.DEFINE_string("use_fp16",False,"train blabla")


FLAGS = flags.FLAGS

##### classes and functions 

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class ds_extended_topic_1_model(object):
    def __init__(self, is_training, config, input_sentence, input_continuous, topic_matrix, initializer_reg, initializer_lda, initializer_int):
        self._input_sentence = input_sentence
        self._input_continuous = input_continuous
        
        batch_size = config['batch_size']
        self._num_steps = num_steps = config['num_steps']
        vocab_size = input_sentence.pad_id #om pad symbool toe te laten
        nb_topics = topic_matrix.get_shape()[0]
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')
        
        with tf.device("/cpu:0"):
            embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config['embedded_size_reg']], dtype=data_type(), initializer = initializer_reg)
            inputs_reg = tf.nn.embedding_lookup(embedding_reg, data)
            # no separate embedding for interpol module
            inputs_int = tf.nn.embedding_lookup(embedding_reg, data)
            embedding_lda = tf.get_variable("embedding_lda", [vocab_size+1, config['embedded_size_lda']], dtype=data_type(), initializer = initializer_lda)
            inputs_lda = tf.nn.embedding_lookup(embedding_lda, data)
            

        with tf.variable_scope('reg_lstm', initializer = initializer_reg) as reg_lstm:
            lstm_cell_reg = tf.nn.rnn_cell.BasicLSTMCell(config['hidden_size_reg'], forget_bias=0.0, state_is_tuple=True)
            cell_reg = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_reg] * config['num_layers_reg'], state_is_tuple=True)

            self._initial_state_reg = cell_reg.zero_state(batch_size, data_type())
            
            outputs_reg, state_reg = tf.nn.dynamic_rnn(cell_reg, inputs_reg, initial_state=self._initial_state_reg, dtype=data_type(), sequence_length=seq_len)
            output_reg = tf.reshape(tf.concat(1, outputs_reg), [-1, config['hidden_size_reg']])
            
        with tf.variable_scope('lda_lstm', initializer = initializer_lda) as lda_lstm:
            lstm_cell_lda = tf.nn.rnn_cell.BasicLSTMCell(config['hidden_size_lda'], forget_bias=0.0, state_is_tuple=True)
            cell_lda = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_lda] * config['num_layers_lda'], state_is_tuple=True)

            self._initial_state_lda = cell_lda.zero_state(batch_size, data_type())
            
            outputs_lda, state_lda = tf.nn.dynamic_rnn(cell_lda, inputs_lda, initial_state=self._initial_state_lda, dtype=data_type(), sequence_length=seq_len)
            output_lda = tf.reshape(tf.concat(1, outputs_lda), [-1, config['hidden_size_lda']])
            
        with tf.variable_scope('int_lstm', initializer = initializer_int) as int_lstm:
            lstm_cell_int = tf.nn.rnn_cell.BasicLSTMCell(config['hidden_size_int'], forget_bias=0.0, state_is_tuple=True)
            cell_int = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_int] * config['num_layers_int'], state_is_tuple=True)

            self._initial_state_int = cell_int.zero_state(batch_size, data_type())
            
            outputs_int, state_int = tf.nn.dynamic_rnn(cell_int, inputs_int, initial_state=self._initial_state_int, dtype=data_type(), sequence_length=seq_len)
            output_int = tf.reshape(tf.concat(1, outputs_int), [-1, config['hidden_size_int']])

        softmax_w_reg = tf.get_variable("softmax_w_reg", [config['hidden_size_reg'], vocab_size], dtype=data_type(), initializer = initializer_reg)
        softmax_b_reg = tf.get_variable("softmax_b_reg", [vocab_size], dtype=data_type(), initializer = initializer_reg)
        
        softmax_w_lda = tf.get_variable("softmax_w_lda", [config['hidden_size_lda'], nb_topics], dtype=data_type(), initializer = initializer_lda)
        softmax_b_lda = tf.get_variable("softmax_b_lda", [nb_topics], dtype=data_type(), initializer = initializer_lda)

        softmax_w_int = tf.get_variable("softmax_w_int", [config['hidden_size_int'], 2], dtype=data_type(), initializer = initializer_int)
        softmax_b_int = tf.get_variable("softmax_b_int", [2], dtype=data_type(), initializer = initializer_int)       
        
        self._cost, self._nb_words_in_batch, self._temp = get_loss_function(output_reg, output_lda, output_int, softmax_w_reg, softmax_w_lda, softmax_w_int, softmax_b_reg, softmax_b_lda, softmax_b_int, labels, topic_matrix, input_sentence, is_training)
        
        self._final_state_reg = state_reg
        self._final_state_lda = state_lda
        self._final_state_int = state_int

        
        
    @property
    def temp(self):
        return self._temp

    @property
    def nb_words_in_batch(self):
        return self._nb_words_in_batch    
        
    @property
    def input_sentence(self):
        return self._input_sentence
    
    @property
    def input_continuous(self):
        return self._input_continuous

    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def data(self):
        return self._data
    
    @property
    def history(self):
        return self._history
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def seq_len(self):
        return self._seq_len

    @property
    def initial_state_reg(self):
        return self._initial_state_reg
    
    @property
    def initial_state_lda(self):
        return self._initial_state_lda
    
    @property
    def initial_state_int(self):
        return self._initial_state_int

    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state_reg(self):
        return self._final_state_reg

    @property
    def final_state_lda(self):
        return self._final_state_lda
    
    @property
    def final_state_int(self):
        return self._final_state_int

    @property
    def lr(self):
        return self._lr
        
    @property
    def train_op(self):
        return self._train_op
    
def get_optimizer(lr):
    if FLAGS.optimizer == "GradDesc":
        return tf.train.GradientDescentOptimizer(lr)
    if FLAGS.optimizer == "Adadelta":
        return tf.train.AdadeltaOptimizer()
    if FLAGS.optimizer == "Adagrad":
        return tf.train.AdagradOptimizer(lr)
    if FLAGS.optimizer == "Momentum":
        return tf.train.MomentumOptimizer(lr,0.33)
    if FLAGS.optimizer == "Adam":
        return tf.train.AdamOptimizer()
    return 0

def get_loss_function(output_reg, output_lda, output_int, softmax_w_reg, softmax_w_lda, softmax_w_int, softmax_b_reg, softmax_b_lda, softmax_b_int, targets, topic_matrix, data, is_training):
    targets = tf.reshape(targets, [-1])
    #if is_training:
    mask = tf.not_equal(targets,[data.pad_id])
    #else:
    #    mask = tf.logical_and(tf.not_equal(targets,[data.pad_id]),tf.not_equal(targets,[data.unk_id]))
    mask2  = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    output_reg = tf.gather(output_reg, mask2)
    output_lda = tf.gather(output_lda, mask2) 
    output_int = tf.gather(output_int, mask2) 
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32))
        
    logits_reg = tf.matmul(output_reg, softmax_w_reg) + softmax_b_reg
    probs_reg = tf.nn.softmax(logits_reg) 
        
    logits_lda = tf.matmul(output_lda, softmax_w_lda) + softmax_b_lda
    probs_topic = tf.nn.softmax(logits_lda) 
    probs_lda = tf.matmul(probs_topic,topic_matrix)
        
        
    logits_int = tf.matmul(output_int, softmax_w_int) + softmax_b_int
    ints = tf.nn.softmax(logits_int) 
        
    probs = tf.slice(ints,[0,0],[-1,1]) * probs_reg + tf.slice(ints,[0,1],[-1,1]) * probs_lda
                
    idx = tf.reshape(targets, [-1])
    idx_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + idx
    y = tf.gather(tf.reshape(probs, [-1]), idx_flattened)  # use flattened indices
    loss = -tf.log(y)
        
    return tf.reduce_sum(loss), nb_words_in_batch, tf.squeeze(probs_topic)


def run_test_epoch(session, model, epoch_nb = 0):
    start_time = time.time()
    costs = 0.0
    iters = 0
    processed_words = 0
    processed_sentences = 0
    word_in_sentence = 0
    pos = 0
    state_reg = session.run(model.initial_state_reg)
    state_lda = session.run(model.initial_state_lda)
    state_int = session.run(model.initial_state_int)


    fetches = {"cost": model.cost,"nb_words_in_batch": model.nb_words_in_batch,"final_state_reg": model.final_state_reg,"final_state_lda": model.final_state_lda, "final_state_int": model.final_state_int, "temp": model.temp}
    
    word_axis = np.zeros((model.input_continuous.epoch_size))
    interpolation = np.zeros((model.input_continuous.epoch_size,512))

    for step in range(model.input_continuous.epoch_size):
        batch_data, batch_labels = model.input_continuous.next_batch()
        batch_seq_len = np.ones(model.input_continuous.batch_size)*model.input_continuous.num_steps
        feed_dict = {}
        feed_dict[model.data] = batch_data
        feed_dict[model.labels] = batch_labels
        feed_dict[model.seq_len] = batch_seq_len

        for i, (c, h) in enumerate(model.initial_state_lda):
            feed_dict[c] = state_lda[i].c
            feed_dict[h] = state_lda[i].h

        for i, (c, h) in enumerate(model.initial_state_reg):
            feed_dict[c] = state_reg[i].c
            feed_dict[h] = state_reg[i].h
            
        for i, (c, h) in enumerate(model.initial_state_int):
            feed_dict[c] = state_int[i].c
            feed_dict[h] = state_int[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state_lda = vals["final_state_lda"]
        state_int = vals["final_state_int"]
        nb_words_in_batch = vals["nb_words_in_batch"]
        #print(vals['temp'])

        if vals['temp'].any():
	    word_in_sentence += 1
            interpolation[step,:] = vals['temp']

        if batch_labels[0,-1] == model.input_continuous.eos_id:
            state_reg = session.run(model.initial_state_reg)
	    word_axis[pos:pos+word_in_sentence] = processed_sentences + np.linspace(0,1,word_in_sentence+1)[:-1]
	    processed_sentences +=1
	    pos += word_in_sentence
	    word_in_sentence = 0
        else:
            state_reg = vals["final_state_reg"]

        costs += cost
        iters += nb_words_in_batch
        processed_words += sum(batch_seq_len)
 	if step % (model.input_continuous.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input_continuous.epoch_size, np.exp(costs / iters),
						 processed_words / (time.time() - start_time)))
 
    print("with(perplexity: %.3f) speed: %.0f wps" % (np.exp(costs / iters),
						 processed_words / (time.time() - start_time)))

    np.savez('interpolation.npz', interpolation=interpolation)
    np.savez('word_axis.npz', word_axis =word_axis)
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.plot(np.arange(model.input_continuous.epoch_size), interpolation,'b', label = 'regular part')
    #plt.plot(np.arange(model.input_continuous.epoch_size), 1-interpolation,'r', label = 'lda part')
    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

    
    #plt.xlabel(r'number of the word in test data')
    #plt.ylabel(r'interpolation factor')
    #plt.title(r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", fontsize=16, color='gray')
    # Make room for the ridiculously large title.
    #plt.subplots_adjust(top=0.8)

    #plt.savefig('tex_demo')
    return np.exp(costs/iters)
 
def main(_):
    print('job started')
    lda_path = os.path.join(FLAGS.data_path, "lda_512_10.ds")
    lda = models.LdaModel.load(lda_path) 
    dict_path = os.path.join(FLAGS.data_path, "dictionary.ds")
    dictionary = corpora.Dictionary.load(dict_path)
    vocab_size = len(dictionary.items())
 
    nb_topics = lda.num_topics
    topic_array = np.zeros((nb_topics, vocab_size))
    for topic_nb in xrange(nb_topics):
        current_topic = lda.get_topic_terms(topic_nb,topn=vocab_size)
        for i in xrange(vocab_size):
            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]

    
    param_np = np.load((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'))
    param_np = param_np['param_train_np']
    
    param1 =  ['num_layers_reg','num_layers_lda','num_layers_int','hidden_size_lda', 'hidden_size_reg', 'hidden_size_int', 'embedded_size_reg', 'embedded_size_lda']
    
    eval_config = {}
    eval_config['batch_size'] = 1
    eval_config['num_steps'] = 1
    for i in range(0,len(param_np)):
        if param_np[i][0] in param1:
            eval_config[param_np[i][0]] = int(param_np[i][1])
    

    with tf.Graph().as_default(): 
        
        topic_matrix = tf.constant(topic_array,dtype=tf.float32)

        with tf.name_scope("test"):
            eval_data_sentence = reader.ds_data_sentence(eval_config['batch_size'], FLAGS.data_path,  FLAGS.eval_name)
            eval_data_continuous = reader.ds_data_continuous(eval_config['batch_size'], eval_config['num_steps'], FLAGS.data_path, FLAGS.eval_name)
            with tf.variable_scope("model"):
                mtest = ds_extended_topic_1_model(is_training=False, config=eval_config, input_sentence = eval_data_sentence, input_continuous = eval_data_continuous, topic_matrix = topic_matrix, initializer_reg = None, initializer_lda = None, initializer_int = None) 

        sv = tf.train.Supervisor(summary_writer=None,save_model_secs=0, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session() as session:
            test_perplexity = run_test_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
                                
if __name__ == "__main__":
    tf.app.run()
