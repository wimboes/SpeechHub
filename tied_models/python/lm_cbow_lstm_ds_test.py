# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np

if 'LD_LIBRARY_PATH' not in os.environ:
        print('hihi')
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
        try:
            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)


import tensorflow as tf
import reader

##### paths

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

##### flags

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","cbow_exp_lstm","test_name")
flags.DEFINE_string("eval_name",'ds.testshort.txt',"eval_name")
flags.DEFINE_integer("top_k",7,"top_k")
flags.DEFINE_integer("neighborhood",7,"neighborhood")

flags.DEFINE_string("loss_function","full_softmax","loss_function")

flags.DEFINE_string("data_path",input_path,"data_path")
flags.DEFINE_string("save_path",output_path,"save_path")
flags.DEFINE_string("use_fp16",False,"train blabla")


FLAGS = flags.FLAGS

##### classes and functions 

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32
    
class ds_cbow_sentence_model(object):

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        self._num_steps = num_steps = config['num_steps']
        vocab_size = input_.pad_id #om pad symbool toe te laten
        hidden_size = config['hidden_size']
        num_history = input_.history_size
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._history = history = tf.placeholder(tf.int32, [batch_size, num_history+num_steps-1], name = 'batch_history')
        self._history_tfidf = history_tfidf = tf.placeholder(tf.int32, [batch_size, num_history+num_steps-1], name = 'batch_history_tfidf')        
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')
        
        with tf.device("/cpu:0"):
            embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config['embedded_size_reg']], dtype=data_type())
            embedding_cbow = tf.get_variable("embedding_cbow", [vocab_size+1, config['embedded_size_cbow']], dtype=data_type())
            inputs_reg = tf.nn.embedding_lookup(embedding_reg, data)
            inputs_cbow = tf.nn.embedding_lookup(embedding_reg, history)

        with tf.variable_scope('cbow'):           
            outputs_cbow = []
            for i in xrange(num_steps):
                slice1 = tf.slice(history,[0,i],[batch_size,num_history])
                slice2 = tf.slice(inputs_cbow,[0,i,0],[batch_size,num_history,config['embedded_size_cbow']])
                
                if config['cbow_combination'] == "mean":
                    mask = tf.cast(tf.logical_and(tf.logical_and(tf.not_equal(slice1,[input_.pad_id]), tf.not_equal(slice1,[input_.unk_id])), tf.logical_and(tf.not_equal(slice1,[input_.bos_id]), tf.not_equal(slice1,[input_.eos_id]))), dtype = data_type())
                    mask1 = tf.pack([mask]*config['embedded_size_cbow'],axis = 2)
                    out = mask1*slice2
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1,1) + 1e-32)
    
                if config['cbow_combination'] == "exp":
                    exp_weights = tf.reverse(tf.constant([[config['embedded_size_cbow']*[config['cbow_exp_decay']**k] for k in range(num_history)] for j in range(batch_size)]),[False,True,False])
                    mask = tf.cast(tf.logical_and(tf.logical_and(tf.not_equal(slice1,[input_.pad_id]), tf.not_equal(slice1,[input_.unk_id])), tf.logical_and(tf.not_equal(slice1,[input_.bos_id]), tf.not_equal(slice1,[input_.eos_id]))), dtype = data_type())
                    mask1 = tf.pack([mask]*config['embedded_size_cbow'],axis = 2)
                    out = mask1*slice2*exp_weights
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1*exp_weights,1) + 1e-32)
                
                if config['cbow_combination'] == "tfidf":
                    tfidf =  tf.slice(history_tfidf,[0,i],[batch_size,num_history])                   
                    out = slice2*tf.expand_dims(tf.cast(tfidf, dtype=data_type()), -1)
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(tf.expand_dims(tf.cast(tfidf, dtype=data_type()), -1),1) + 1e-32)
  
    
                outputs_cbow.append(comb_)
            output_cbow_lstm = tf.reshape(tf.concat(1, outputs_cbow), [batch_size,num_steps, config['embedded_size_cbow']])

            normed_embedding_cbow = tf.nn.l2_normalize(embedding_cbow, dim=1)
            normed_array = tf.nn.l2_normalize(output_cbow_lstm, dim=1)

            cosine_similarity = tf.matmul(tf.squeeze(normed_array), tf.transpose(normed_embedding_cbow, [1, 0]))
            _, closest_ids = tf.nn.top_k(cosine_similarity, k=FLAGS.neighborhood, sorted=True)
            self._temp5 = closest_ids


        with tf.variable_scope('lstm_lstm'):

            lstm_cell_lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_lstm] * config['num_layers'], state_is_tuple=True)

            self._initial_state_lstm = cell_lstm.zero_state(batch_size, data_type())
            
            inputs_lstm = tf.concat(2,[inputs_reg, output_cbow_lstm])
            outputs_lstm, state_lstm = tf.nn.dynamic_rnn(cell_lstm, inputs_lstm, initial_state=self._initial_state_lstm, dtype=data_type(), sequence_length=seq_len)
            output_LSTM_lstm = tf.reshape(tf.concat(1, outputs_lstm), [-1, hidden_size])
            output_lstm = output_LSTM_lstm

            
        softmax_w_lstm = tf.get_variable("softmax_w_lstm", [hidden_size, vocab_size], dtype=data_type())
        softmax_b_lstm = tf.get_variable("softmax_b_lstm", [vocab_size], dtype=data_type())            
          

        self._cost, self._nb_words_in_batch = get_loss_function(output_lstm, softmax_w_lstm, softmax_b_lstm, labels, input_, is_training)
        self._temp1,self._temp2,self._temp3 = get_N_most_probable_words(output_lstm, softmax_w_lstm, softmax_b_lstm, labels, data, input_)
        _,_,self._temp4 = get_probability(output_lstm, softmax_w_lstm, softmax_b_lstm, labels, data, input_)
        
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
    def temp5(self):
        return self._temp5

    @property
    def input(self):
        return self._input

    @property
    def initial_state_soft(self):
        return self._initial_state_soft
        
    @property
    def initial_state_lstm(self):
        return self._initial_state_lstm
        
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
    def history_tfidf(self):
        return self._history_tfidf 
        
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


             
    
def get_N_most_probable_words(output, softmax_w, softmax_b, targets, data, model):
    targets = tf.reshape(targets, [-1])
    data = tf.reshape(data, [-1])
    mask = tf.not_equal(targets,[model.pad_id]) 
    mask2 = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    data = tf.gather(data, mask2)
    output = tf.gather(output, mask2)
    
    logits = tf.matmul(output, softmax_w) + softmax_b
    _, indeces = tf.nn.top_k(logits, k=FLAGS.top_k, sorted=True)

    return data, targets, indeces

def get_probability(output, softmax_w, softmax_b, targets, data, model):
    targets = tf.reshape(targets, [-1])
    data = tf.reshape(data, [-1])
    mask = tf.not_equal(targets,[model.pad_id]) 
    mask2 = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    data = tf.gather(data, mask2)
    output = tf.gather(output, mask2)
    
    logits = tf.matmul(output, softmax_w) + softmax_b
    probability = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)

    return data, targets, probability

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

def run_epoch(session, model, cost=None, eval_op=None):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    processed_words = 0
    
    reverse_dict = {v: k for k, v in model.input.word_to_id.iteritems()}
    reverse_dict[model.input.pad_id] = 'PAD'
    
    fetches = {"cost": model.cost,"nb_words_in_batch": model.nb_words_in_batch, "temp1" :model.temp1, "temp2" :model.temp2, "temp3" :model.temp3, "temp4" :model.temp4, "temp5":model.temp5}

    if (os.path.exists((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/eval' +'.txt'))):
        os.remove(FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/eval' +'.txt')

    with open((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run) + '/eval' +'.txt'), "w") as f, open((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run) + '/domain_eval' +'.txt'), "w") as g:
        for step in range(model.input.epoch_size):
            batch_data, batch_history, batch_history_tfidf, batch_labels, batch_seq_len = model.input.next_batch(model.num_steps)
            feed_dict = {}
            feed_dict[model.data] = batch_data
            feed_dict[model.history] = batch_history
            feed_dict[model.history_tfidf] = batch_history_tfidf
            feed_dict[model.labels] = batch_labels
            feed_dict[model.seq_len] = batch_seq_len
    
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            nb_words_in_batch = vals["nb_words_in_batch"]

            data = vals["temp1"]
            labels = vals["temp2"]
            top_k = vals["temp3"]
            prob = vals['temp4']
            for i in xrange(len(data)):
                f.write("{:<15}".format(reverse_dict[data[i]].encode('utf-8')))
                f.write(" | ")
                f.write("{:<15}".format(reverse_dict[labels[i]].encode('utf-8')))
                f.write(" | ")
                f.write("{:<15}".format(prob[i]))
                f.write(" -> ")
                f.write("{:<15}".format(('%e'% np.exp(-prob[i]))))
                f.write(" | ")
                for j in xrange(len(top_k[i])):
                    f.write("{:<15}".format(reverse_dict[top_k[i][j]].encode('utf-8')))
                f.write("\n")

            domain_info = vals["temp5"]
            for i in xrange(len(data)):
                g.write("{:<15}".format(reverse_dict[data[i]].encode('utf-8')))
                g.write("| ")
                for j in xrange(len(domain_info[i])):
                    g.write("{:<15}".format(reverse_dict[domain_info[i][j]].encode('utf-8')))
                g.write("\n")   

            costs += cost
            iters += nb_words_in_batch
            processed_words += sum(batch_seq_len)
    
            if step % (model.input.epoch_size // 10) == 0:
                print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
    						 processed_words / (time.time() - start_time))) 
    return np.exp(costs/iters)

 
def main(_):
    print('Eval job started')
    
    param_np = np.load((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'))
    param_np = param_np['param_train_np']
    
    param1 =  ['num_layers', 'hidden_size', 'embedded_size_reg', 'embedded_size_cbow', 'num_history']
    param2 = ['cbow_exp_decay']
    param3 =  ['cbow_position', 'cbow_combination']
    
    eval_config = {}
    eval_config['batch_size'] = 1
    for i in range(0,len(param_np)):
        if param_np[i][0] in param1:
            eval_config[param_np[i][0]] = int(param_np[i][1])
        elif param_np[i][0] in param2:
            eval_config[param_np[i][0]] = float(param_np[i][1])
        elif param_np[i][0] in param3:
            eval_config[param_np[i][0]] = param_np[i][1] 
    
    with tf.Graph().as_default():
        with tf.name_scope("test"):
            eval_data = reader.ds_data_sentence_with_history(eval_config['batch_size'], eval_config['num_history'], FLAGS.data_path, FLAGS.eval_name)
            eval_config['num_steps'] = eval_data.longest_sentence
            with tf.variable_scope("model"):
                mtest = ds_cbow_sentence_model(is_training=False, config=eval_config, input_=eval_data)
		
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth=True

        sv = tf.train.Supervisor(summary_writer=None,save_model_secs=300, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session(config=conf) as session:
            test_perplexity=  run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
    
    print('done')
if __name__ == "__main__":
    tf.app.run()
