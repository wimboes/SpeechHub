# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
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
import shutil
from gensim import corpora, models


python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input_n_best/original_n-best')
data_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
output_path = os.path.join(general_path,'output')

# set data and save path

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","topic","test_name")

flags.DEFINE_string("name","n-best-topic","name")

flags.DEFINE_string("input_path", input_path, "data_path")
flags.DEFINE_string("model_name", "n-best", "model_name")
flags.DEFINE_string("data_path", data_path, "data_path")
flags.DEFINE_string("save_path", output_path, "save_path")
flags.DEFINE_bool("use_fp16", False, "train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

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

        softmax_w_reg = tf.get_variable("softmax_w_reg", [config['hidden_size_reg'], vocab_size], dtype=data_type(), initializer = initializer_reg)
        softmax_b_reg = tf.get_variable("softmax_b_reg", [vocab_size], dtype=data_type(), initializer = initializer_reg)
        
        softmax_w_lda = tf.get_variable("softmax_w_lda", [config['hidden_size_lda'], nb_topics], dtype=data_type(), initializer = initializer_lda)
        softmax_b_lda = tf.get_variable("softmax_b_lda", [nb_topics], dtype=data_type(), initializer = initializer_lda)
        
        
        self._interpol = tf.get_variable("interpol", [], dtype=data_type())
    
        self._cost, self._nb_words_in_batch = get_loss_function(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, self._interpol, labels, topic_matrix, input_continuous, is_training)
        
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
    def temp5(self):
        return self._temp5
        
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
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32)) + 1e-32
        
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


def run_test_epoch(session, model, times, epoch_nb = 0):
    """Runs the model on the given data."""
    costs = 0.0
    iters = 0
    state_reg = session.run(model.initial_state_reg)
    state_lda = session.run(model.initial_state_lda)

    
    fetches = {"cost": model.cost, 'nb_words_in_batch': model.nb_words_in_batch, "final_state_reg": model.final_state_reg,"final_state_lda": model.final_state_lda}

    for step in range(times):
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

        costs += cost
        iters += nb_words_in_batch

    return np.exp(costs/iters)    
 
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
    
    lda_path = os.path.join(FLAGS.data_path, "lda_512_10.ds")
    lda = models.LdaModel.load(lda_path) 
    dict_path = os.path.join(FLAGS.data_path, "dictionary.ds")
    dictionary = corpora.Dictionary.load(dict_path)
    word_to_id = dict()
    for (wordid,word) in dictionary.iteritems():
            word_to_id[word] = wordid   
    vocab_size = len(dictionary.items())
    word_to_id_set = set(word_to_id.keys())

 
    nb_topics = lda.num_topics
    topic_array = np.zeros((nb_topics, vocab_size))
    for topic_nb in xrange(nb_topics):
        current_topic = lda.get_topic_terms(topic_nb,topn=vocab_size)
        for i in xrange(vocab_size):
            topic_array[topic_nb,current_topic[i][0]] = current_topic[i][1]
    
    param_np = np.load((model_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'))
    param_np = param_np['param_train_np']
    
    param1 =  ['num_layers_reg','num_layers_lda','hidden_size_lda', 'hidden_size_reg', 'embedded_size_reg', 'embedded_size_lda']
    
    eval_config = {}
    eval_config['batch_size'] = 1
    eval_config['num_steps'] = 1
    for i in range(0,len(param_np)):
        if param_np[i][0] in param1:
            eval_config[param_np[i][0]] = int(param_np[i][1])
    
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
        for j in range(fv_files_amount[i]+1):
            name_file = fv_files[i] + '.0.' + str(j) + '.10000-best.txt'
            output_file = os.path.join(output_dir, name_file)

            input_file = os.path.join(FLAGS.input_path, name_file)
            sentences = []
            #read all sentence hypotheses	
            with open(input_file,'r') as f:
                for k in range(1000): 
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
                    len_sentences.append(len_sentences[-1] + best_sentence_history_len + len(new_sen) - 1)     #setence level           
                    new_sen = ' '.join(new_sen)
                    for best_sen in best_sentence_history: g.write(best_sen +'\n')
                    g.write(new_sen +'\n') 	
            
            #compute perplexity
            ppl = []
            with tf.Graph().as_default():
                topic_matrix = tf.constant(topic_array,dtype=tf.float32)
                    
                with tf.name_scope("test"):
                    eval_data = reader.ds_data_continuous(eval_config['batch_size'], eval_config['num_steps'], FLAGS.data_path,output_dir,'hypothesis_file' + str(j) +'.txt')
                    with tf.variable_scope("model"):
                        mtest =  ds_topic_model(is_training=False, config=eval_config, input_sentence = None, input_continuous = eval_data, topic_matrix = topic_matrix, initializer_reg = None, initializer_lda = None)

                sv = tf.train.Supervisor(summary_writer=None, save_model_secs=0, logdir=model_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
                with sv.managed_session() as session:
                    for k in range(len(sentences)):  
                        test_perplexity=  run_test_epoch(session, mtest, len_sentences[k+1]-len_sentences[k])
                        ppl.append(-test_perplexity)
                        print("hypothesis %d with PPL %.3f" % (k,test_perplexity))



            vals = np.array(ppl) + np.array(accoustic_score)	
            sort_index = np.argsort(vals)[::-1]
            
            #best sentence:
            sen = sentences[sort_index[0]].split()
            new_sen = []
            for k in range(len(sen)): 
                if sen[k] in word_to_id.keys():
                    new_sen.append(sen[k].encode('utf-8'))
            best_sentence_history_len += len(new_sen) -1
            new_sen = ' '.join(new_sen)
            best_sentence_history.append(new_sen)


            sentences2 = []
            for k in sort_index:
                sen = sentences[k].split()
                sen[1] = str(ppl[k])
                sen = ' '.join(sen)
                sentences2.append(sen.encode('utf-8'))
            
            with open(output_file,'w') as h:
                for sentence in sentences2: h.write(sentence +'\n')
                    
    os.system('python compute_WER.py  --n_best ' +  output_dir + ' --name ' + FLAGS.name + '_WER')


                    
if __name__ == "__main__":
    tf.app.run()
