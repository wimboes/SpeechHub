##### comments

# first run transform_ds.py and lda_generator_ds.py before running this file

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

##### paths

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

##### flags

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","topic","test_name")
flags.DEFINE_string("eval_name",'ds.valid.txt',"eval_name")
flags.DEFINE_integer("interpol",0.5,"interpol") #0 is reg, 1 is lda

flags.DEFINE_string("loss_function","full_softmax","loss_function")

flags.DEFINE_string("data_path",input_path,"data_path")
flags.DEFINE_string("save_path",output_path,"save_path")
flags.DEFINE_string("use_fp16",False,"train blabla")


FLAGS = flags.FLAGS

##### classes and functions 

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class ds_topic_model(object):
    def __init__(self, is_training, config, input_sentence, input_continuous, topic_matrix, initializer_reg, initializer_lda):
        self._input_sentence = input_sentence
        self._input_continuous = input_continuous
        
        batch_size = config['batch_size']
        self._num_steps = num_steps = config['num_steps']
        vocab_size = input_continuous.pad_id #om pad symbool toe te laten
        nb_topics = topic_matrix.get_shape()[0]
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')
        
        with tf.device("/cpu:0"):
            embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config['embedded_size_reg']], dtype=data_type(), initializer = initializer_reg)
            inputs_reg = tf.nn.embedding_lookup(embedding_reg, data)
            embedding_lda = tf.get_variable("embedding_lda", [vocab_size+1, config['embedded_size_lda']], dtype=data_type(), initializer = initializer_lda)
            inputs_lda = tf.nn.embedding_lookup(embedding_reg, data)


        with tf.variable_scope('reg_lstm', initializer = initializer_reg):
            lstm_cell_reg = tf.nn.rnn_cell.BasicLSTMCell(config['hidden_size_reg'], forget_bias=0.0, state_is_tuple=True)
            cell_reg = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_reg] * config['num_layers_reg'], state_is_tuple=True)

            self._initial_state_reg = cell_reg.zero_state(batch_size, data_type())
            
            outputs_reg, state_reg = tf.nn.dynamic_rnn(cell_reg, inputs_reg, initial_state=self._initial_state_reg, dtype=data_type(), sequence_length=seq_len)
            output_reg = tf.reshape(tf.concat(1, outputs_reg), [-1, config['hidden_size_reg']])
            
        with tf.variable_scope('lda_lstm', initializer = initializer_lda):
            lstm_cell_lda = tf.nn.rnn_cell.BasicLSTMCell(config['hidden_size_lda'], forget_bias=0.0, state_is_tuple=True)
            cell_lda = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_lda] * config['num_layers_lda'], state_is_tuple=True)

            self._initial_state_lda = cell_lda.zero_state(batch_size, data_type())
            
            outputs_lda, state_lda = tf.nn.dynamic_rnn(cell_lda, inputs_lda, initial_state=self._initial_state_lda, dtype=data_type(), sequence_length=seq_len)
            output_lda = tf.reshape(tf.concat(1, outputs_lda), [-1, config['hidden_size_lda']])

        softmax_w_reg = tf.get_variable("softmax_w_reg", [config['hidden_size_reg'], vocab_size], dtype=data_type(), initializer = initializer_reg)
        softmax_b_reg = tf.get_variable("softmax_b_reg", [vocab_size], dtype=data_type(), initializer = initializer_reg)
        
        softmax_w_lda = tf.get_variable("softmax_w_lda", [config['hidden_size_lda'], nb_topics], dtype=data_type(), initializer = initializer_lda)
        softmax_b_lda = tf.get_variable("softmax_b_lda", [nb_topics], dtype=data_type(), initializer = initializer_lda)
        
    
        self._cost, self._nb_words_in_batch = get_loss_function(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, FLAGS.interpol, labels, topic_matrix, input_continuous, is_training)
        self._temp1,self._temp2,self._temp3 = get_probability(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, FLAGS.interpol, labels, topic_matrix, data, input_continuous)
        
        self._final_state_reg = state_reg
        self._final_state_lda = state_lda
    
    @property
    def nb_words_in_batch(self):
        return self._nb_words_in_batch    

    @property
    def temp1(self):
        return self._temp1
    
    @property
    def temp2(self):
        return self._temp2

    @property
    def temp3(self):
        return self._temp3

    @property
    def temp4(self):
        return self._temp4
        
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
    def cost(self):
        return self._cost
    
    @property
    def final_state_reg(self):
        return self._final_state_reg

    @property
    def final_state_lda(self):
        return self._final_state_lda

    
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
    
def get_probability(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, interpol, targets, topic_matrix, data, model):
    targets = tf.reshape(targets, [-1])
    data = tf.reshape(data, [-1])
    mask = tf.not_equal(targets,[model.pad_id])
    mask2  = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    data = tf.gather(data, mask2)
    output_reg = tf.gather(output_reg, mask2)
    output_lda = tf.gather(output_lda, mask2) 

        
    if FLAGS.loss_function == "full_softmax":        
        logits_reg = tf.matmul(output_reg, softmax_w_reg) + softmax_b_reg
        probs_reg = tf.nn.softmax(logits_reg) 
        
        logits_lda = tf.matmul(output_lda, softmax_w_lda) + softmax_b_lda
        probs_topic = tf.nn.softmax(logits_lda) 
        probs_lda = tf.matmul(probs_topic,topic_matrix)
        
        probs = (1-interpol)*probs_reg + (interpol)*probs_lda
                
        idx = tf.reshape(targets, [-1])
        idx_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + idx
        probability = tf.gather(tf.reshape(probs, [-1]), idx_flattened)  # use flattened indices

        return data, targets, probability

def get_loss_function(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, interpol, targets, topic_matrix, model, is_training):
    targets = tf.reshape(targets, [-1])
    if is_training:
        mask = tf.not_equal(targets,[model.pad_id])
    else:
        mask = tf.logical_and(tf.not_equal(targets,[model.pad_id]),tf.not_equal(targets,[model.unk_id]))
    mask2  = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    output_reg = tf.gather(output_reg, mask2)
    output_lda = tf.gather(output_lda, mask2) 
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32))
        
    if FLAGS.loss_function == "full_softmax":        
        logits_reg = tf.matmul(output_reg, softmax_w_reg) + softmax_b_reg
        probs_reg = tf.nn.softmax(logits_reg) 
        
        logits_lda = tf.matmul(output_lda, softmax_w_lda) + softmax_b_lda
        probs_topic = tf.nn.softmax(logits_lda) 
        probs_lda = tf.matmul(probs_topic,topic_matrix)
        
        probs = (1-interpol)*probs_reg + (interpol)*probs_lda
                
        idx = tf.reshape(targets, [-1])
        idx_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + idx
        y = tf.gather(tf.reshape(probs, [-1]), idx_flattened)  # use flattened indices
        loss = -tf.log(y)
        return tf.reduce_sum(loss), nb_words_in_batch


def run_test_epoch(session, model, epoch_nb = 0):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state_reg = session.run(model.initial_state_reg)
    state_lda = session.run(model.initial_state_lda)
    processed_words = 0
    sentence = []
    probs = []
    unk_mask = []
    nb_words = 0
    nb_unks = 0
    nb_sentences= 0
    tot_logprob = 0
    
    reverse_dict = {v: k for k, v in model.input_continuous.word_to_id.iteritems()}
    reverse_dict[model.input_continuous.pad_id] = 'PAD'
    
    fetches = {"cost": model.cost,"nb_words_in_batch": model.nb_words_in_batch,  "final_state_reg": model.final_state_reg,"final_state_lda": model.final_state_lda, "temp1" :model.temp1, "temp2" :model.temp2, "temp3" :model.temp3}

    if (os.path.exists((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/eval' +'.txt'))):
        os.remove(FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/eval' +'.txt')

    with open((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run) + '/interpol' + str(FLAGS.interpol)+'.txt'), "w") as f:
        f.write('\n')        
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
    
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state_lda = vals["final_state_lda"]
            nb_words_in_batch = vals["nb_words_in_batch"]

            if batch_labels[0,0] == model.input_continuous.eos_id:
                state_reg = session.run(model.initial_state_reg)
            else:
                state_reg = vals["final_state_reg"]

            data = vals["temp1"]
            prob = vals["temp3"]
            if data == model.input_continuous.unk_id:
                sentence.append(reverse_dict[data[0]].encode('utf-8'))
                probs.append(prob[0])
                unk_mask.append(1)
                
            elif data != model.input_continuous.eos_id:
                sentence.append(reverse_dict[data[0]].encode('utf-8'))
                probs.append(prob[0])
                unk_mask.append(0)
            else:
                sentence.append(reverse_dict[data[0]].encode('utf-8'))
                probs.append(prob[0])
                unk_mask.append(0)
                logprob = sum([np.log10(probs[i]) for i in range(1,len(probs)) if unk_mask[i]==0])
                unks = sum(unk_mask)
                ppl = 10**(-logprob/(len(sentence)-1- unks))
                ppl1 = 10**(-logprob/(len(sentence)-2-unks))
                f.write( ' '.join(sentence[1:-1]) + ' \n')
                for i in range(1,len(sentence)):
                    f.write('\tp( ')
                    f.write(sentence[i])
                    f.write(' | ')
                    f.write(sentence[i-1])                    
                    if unk_mask[i] ==0: 
                        f.write(' ) \t= ['+ FLAGS.test_name + '] ') 
                    else: 
                        f.write(' ) \t= [OOV] ')
                    f.write('%e'% probs[i])
                    f.write(' [ '+ str(np.log10(probs[i])) +' ] ')
                    f.write('\n')
                f.write('1 sentences, '+ str(len(sentence)-2) + ' words, '+ str(unks) + ' OOVs \n') 
                f.write('0 zeroprobs, logprob='+ str(logprob) + ' ppl=' + str(ppl) + ' ppl1 ='+ str(ppl1) +' \n') 
                f.write('\n')
                nb_words += len(sentence)-2
                nb_unks += unks
                tot_logprob += logprob
                nb_sentences += 1
                sentence = []
                probs = []
                unk_mask = []
            
            
            costs += cost
            iters += nb_words_in_batch
            processed_words += sum(batch_seq_len)
    
            if step % (model.input_continuous.epoch_size // 10) == 0:
                print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input_continuous.epoch_size, np.exp(costs / iters),
    						 processed_words / (time.time() - start_time))) 
        f.write('file ' + FLAGS.data_path + '/intepol' + str(FLAGS.interpol)+': '+ str(nb_sentences) + ' sentences, ' + str(nb_words) + ' words, ' + str(nb_unks) + ' OOVs \n')   
        f.write('0 zeroprobs, logprob='+ str(tot_logprob) + ' ppl=' + str(10**(-tot_logprob/(nb_words+nb_sentences-nb_unks))) + ' ppl1 ='+ str(10**(-tot_logprob/(nb_words-nb_unks))) +' \n') 
    return np.exp(costs/iters)    
 
def main(_):
    print('Eval job started')
    
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
    
    param1 =  ['num_layers_reg','num_layers_lda','hidden_size_lda', 'hidden_size_reg', 'embedded_size_reg', 'embedded_size_lda']
    
    eval_config = {}
    eval_config['batch_size'] = 1
    eval_config['num_steps'] = 1
    for i in range(0,len(param_np)):
        if param_np[i][0] in param1:
            eval_config[param_np[i][0]] = int(param_np[i][1])
    
    with tf.Graph().as_default():
        topic_matrix = tf.constant(topic_array,dtype=tf.float32)
        
        with tf.name_scope("test"):
            eval_data = reader.ds_data_continuous(eval_config['batch_size'], eval_config['num_steps'], FLAGS.data_path, FLAGS.eval_name)
            with tf.variable_scope("model"):
                mtest =  ds_topic_model(is_training=False, config=eval_config, input_sentence = None, input_continuous = eval_data, topic_matrix = topic_matrix, initializer_reg = None, initializer_lda = None)
    
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth=True

        sv = tf.train.Supervisor(summary_writer=None,save_model_secs=300, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session(config=conf) as session:
            test_perplexity=  run_test_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
    print('done')
                    
if __name__ == "__main__":
    tf.app.run()
