# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
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
import shutil
from gensim import corpora


python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input_n_best/original_n-best')
data_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

# set data and save path

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","cbow_exp_lstm","test_name")

flags.DEFINE_string("name","pretrained_untied_n_best_cbow_exp_lstm_small","name")

flags.DEFINE_string("input_path", input_path, "data_path")
flags.DEFINE_string("model_name", "pretrained_untied_models_small", "model_name")
flags.DEFINE_string("data_path", data_path, "data_path")
flags.DEFINE_string("save_path", output_path, "save_path")
flags.DEFINE_bool("use_fp16", False, "train using 16-bit floats instead of 32bit floats")


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
            inputs_cbow = tf.nn.embedding_lookup(embedding_cbow, history)

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
        
    @property
    def nb_words_in_batch(self):
        return self._nb_words_in_batch    
    
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

    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
    return tf.reduce_sum(loss), nb_words_in_batch

def run_epoch(session, model, times, cost=None, eval_op=None):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    model.input.reset_history()

    fetches = {"cost": model.cost,"nb_words_in_batch": model.nb_words_in_batch}


    for step in range(times):
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

        costs += cost
        iters += nb_words_in_batch
    
    return -costs*np.log10(np.exp(1))
    
def find_n_best_lists(n_best):
    n_best_files = os.listdir(n_best)
    n_best_files.sort()
    fv_files = []
    fv_files_amount = []
    for file in n_best_files:
        if not file.split('.')[0] in fv_files:
            fv_files.append(file.split('.')[0])
            fv_files_amount.append(int(file.split('.')[2]))
        else:
            if fv_files_amount[-1] < int(file.split('.')[2]):
                fv_files_amount[-1] = int(file.split('.')[2])
    
    return fv_files, fv_files_amount

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
    model_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],FLAGS.model_name +'/output')

    dict_path = os.path.join(FLAGS.data_path, "dictionary.ds")
    dictionary = corpora.Dictionary.load(dict_path)
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
            word_to_id[word] = wordid
    word_to_id_set = set(word_to_id.keys())

    param_np = np.load((model_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'))
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
    
    output_dir = os.path.join(FLAGS.save_path, FLAGS.model_name, FLAGS.name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        remove(output_dir)
        os.mkdir(output_dir)
        

    fv_files, fv_files_amount = find_n_best_lists(FLAGS.input_path)
    for i in range(len(fv_files)):
        print(fv_files[i])
        best_sentence_history = []
        best_sentence_history_len = 0
        ppl_so_far = 0
        for j in range(fv_files_amount[i]+1):
            name_file = fv_files[i] + '.0.' + str(j) + '.10000-best.txt'
            output_file = os.path.join(output_dir, name_file)

            input_file = os.path.join(FLAGS.input_path, name_file)
            sentences = []
            #read all sentence hypotheses	
            with open(input_file,'r') as f:
                for k in range(100): 
                    line = f.readline().decode('utf-8')
                    if line == '':
                        break
                    sentences.append(line)	
	
            #make sentence files that later needs to be grades
            accoustic_score = [float(sentence.split()[0]) for sentence in sentences]	
            
            len_sentences = [0]
            with open(os.path.join(output_dir,'hypothesis_file' + str(j) +'.txt'),'w') as g:
                for k in range(len(sentences)):
                    sen = sentences[k].split()
                    new_sen = []
                    for l in range(len(sen)): 
                        if sen[l] in word_to_id_set:
                            new_sen.append(sen[l].encode('utf-8'))
                    len_sentences.append(len_sentences[-1] + best_sentence_history_len + 1)     #setence level           
                    new_sen = ' '.join(new_sen)
                    for best_sen in best_sentence_history: g.write(best_sen +'\n')
                    g.write(new_sen +'\n') 	
            
            #compute perplexity
            ppl = []
            with tf.Graph().as_default():
                with tf.name_scope("test"):
                    
                    eval_data = reader.ds_data_sentence_with_history(eval_config['batch_size'], eval_config['num_history'], FLAGS.data_path, output_dir,'hypothesis_file' + str(j)+'.txt')
                    eval_config['num_steps'] = eval_data.longest_sentence
                    with tf.variable_scope("model"):
                        mtest = ds_cbow_sentence_model(is_training=False, config=eval_config, input_=eval_data)

                    sv = tf.train.Supervisor(summary_writer=None, save_model_secs=0, logdir=model_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
                    with sv.managed_session() as session:
                        for k in range(len(sentences)):  
                            test_perplexity=  run_epoch(session, mtest, len_sentences[k+1]-len_sentences[k])
                            ppl.append(test_perplexity-ppl_so_far)
                            #print("hypothesis %d with PPL %.3f" % (k,test_perplexity-ppl_so_far))



            vals = np.array(ppl) + np.array(accoustic_score)	
            sort_index = np.argsort(vals)[::-1]
            
            #best sentence:
            sen = sentences[sort_index[0]].split()
            new_sen = []
            for k in range(len(sen)): 
                if sen[k] in word_to_id.keys():
                    new_sen.append(sen[k].encode('utf-8'))
            new_sen = ' '.join(new_sen)
            best_sentence_history.append(new_sen)
            best_sentence_history_len += 1
            ppl_so_far += ppl[sort_index[0]]

            sentences2 = []
            for k in sort_index:
                sen = sentences[k].split()
                sen[1] = str(ppl[k])
                sen = ' '.join(sen)
                sentences2.append(sen.encode('utf-8'))
            
            with open(output_file,'w') as h:
                for sentence in sentences2: h.write(sentence +'\n')
                    
    os.system('python compute_WER.py  --n_best ' +  output_dir + ' --name ' + FLAGS.name + '_WER')


    print('done')
    
if __name__ == "__main__":
    tf.app.run()
