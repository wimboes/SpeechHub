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

### regular

flags.DEFINE_float("init_scale", 0.05, "init_scale")
flags.DEFINE_float("learning_rate", 1, "learning_rate")
flags.DEFINE_float("max_grad_norm", 5, "max_grad_norm")
flags.DEFINE_integer("num_layers", 1, "num_layers")
flags.DEFINE_integer("num_history", 80, "num_history")
flags.DEFINE_float("cbow_exp_decay", 0.9, "cbow_exp_decay")
flags.DEFINE_integer("hidden_size", 512, "hidden_size")
flags.DEFINE_integer("max_epoch", 3, "max_epoch")
flags.DEFINE_integer("max_max_epoch", 3, "max_max_epoch")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
flags.DEFINE_float("lr_decay", 0.8, "lr_decay")
flags.DEFINE_integer("embedded_size_reg", 128, "embedded_size_reg")
flags.DEFINE_integer("embedded_size_cbow", 128, "embedded_size_cbow")

### general

flags.DEFINE_integer("batch_size", 50, "batch_size")
flags.DEFINE_integer("num_steps", 50, "num_steps")
flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","cbow_test_lstm","test_name")
flags.DEFINE_string("data_path",input_path,"data_path")
flags.DEFINE_string("save_path",output_path,"save_path")
flags.DEFINE_string("use_fp16",False,"train blabla")
flags.DEFINE_string("loss_function","full_softmax","loss_function")
flags.DEFINE_string("optimizer","Adagrad","optimizer")
flags.DEFINE_string("combination","tfidf","combination")
flags.DEFINE_string("position","lstm","position")
flags.DEFINE_string("pretrained", "yes", "pretrained")

FLAGS = flags.FLAGS

##### classes and functions 

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32
    
class ds_cbow_sentence_model(object):

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        self._num_steps = num_steps = config.num_steps
        vocab_size = input_.pad_id #om pad symbool toe te laten
        hidden_size = config.hidden_size
        num_history = input_.history_size
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._history = history = tf.placeholder(tf.int32, [batch_size, num_history+num_steps-1], name = 'batch_history')
        self._history_tfidf = history_tfidf = tf.placeholder(tf.int32, [batch_size, num_history+num_steps-1], name = 'batch_history_tfidf')
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')
        
	if FLAGS.pretrained == "yes":
            input_path = os.path.join(os.path.split(os.path.split(python_path)[0])[0],'input')
	    embedding_np= np.load(os.path.join(input_path,"embedding_128.npy"))	
	    with tf.device("/cpu:0"):
                embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config.embedded_size_reg], initializer=tf.constant_initializer(embedding_np),  dtype=data_type())
                embedding_cbow = tf.get_variable("embedding_cbow", [vocab_size+1, config.embedded_size_cbow], initializer=tf.constant_initializer(embedding_np),  dtype=data_type())
                inputs_reg = tf.nn.embedding_lookup(embedding_reg, data)
                inputs_cbow = tf.nn.embedding_lookup(embedding_cbow, history)
	else:
            with tf.device("/cpu:0"):
                embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config.embedded_size_reg], dtype=data_type())
                embedding_cbow = tf.get_variable("embedding_cbow", [vocab_size+1, config.embedded_size_cbow],  dtype=data_type())
                inputs_reg = tf.nn.embedding_lookup(embedding_reg, data)
                inputs_cbow = tf.nn.embedding_lookup(embedding_cbow, history)

        if is_training and config.keep_prob < 1:
            inputs_reg = tf.nn.dropout(inputs_reg, config.keep_prob)
            inputs_cbow = tf.nn.dropout(inputs_cbow, config.keep_prob)

        with tf.variable_scope('cbow') as cbow:           
            outputs_cbow = []
            for i in xrange(num_steps):
                slice1 = tf.slice(history,[0,i],[batch_size,num_history])
                slice2 = tf.slice(inputs_cbow,[0,i,0],[batch_size,num_history,config.embedded_size_cbow])
                
                if FLAGS.combination == "mean":
                    mask = tf.cast(tf.logical_and(tf.logical_and(tf.not_equal(slice1,[input_.pad_id]), tf.not_equal(slice1,[input_.unk_id])), tf.logical_and(tf.not_equal(slice1,[input_.bos_id]), tf.not_equal(slice1,[input_.eos_id]))), dtype = data_type())
                    mask1 = tf.pack([mask]*config.embedded_size_cbow,axis = 2)
                    out = mask1*slice2
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1,1) + 1e-32)
    
                if FLAGS.combination == "exp":
                    exp_weights = tf.reverse(tf.constant([[config.embedded_size_cbow*[config.cbow_exp_decay**k] for k in range(num_history)] for j in range(batch_size)]),[False,True,False])
                    mask = tf.cast(tf.logical_and(tf.logical_and(tf.not_equal(slice1,[input_.pad_id]), tf.not_equal(slice1,[input_.unk_id])), tf.logical_and(tf.not_equal(slice1,[input_.bos_id]), tf.not_equal(slice1,[input_.eos_id]))), dtype = data_type())
                    mask1 = tf.pack([mask]*config.embedded_size_cbow,axis = 2)
                    out = mask1*slice2*exp_weights
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1*exp_weights,1) + 1e-32)
                    
                if FLAGS.combination == "tfidf":
                    tfidf =  tf.slice(history_tfidf,[0,i],[batch_size,num_history])                   
                    out = slice2*tf.expand_dims(tf.cast(tfidf, dtype=data_type()), -1)
                    comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(tf.expand_dims(tf.cast(tfidf, dtype=data_type()), -1),1) + 1e-32)
    
                outputs_cbow.append(comb_)
            output_cbow_lstm = tf.reshape(tf.concat(1, outputs_cbow), [batch_size,num_steps, config.embedded_size_cbow])
        
        with tf.variable_scope('lstm_lstm') as lstm_lstm:

            lstm_cell_lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                lstm_cell_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_lstm, output_keep_prob=config.keep_prob)
            cell_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_lstm] * config.num_layers, state_is_tuple=True)

            self._initial_state_lstm = cell_lstm.zero_state(batch_size, data_type())
            
            inputs_lstm = tf.concat(2,[inputs_reg, output_cbow_lstm])
            outputs_lstm, state_lstm = tf.nn.dynamic_rnn(cell_lstm, inputs_lstm, initial_state=self._initial_state_lstm, dtype=data_type(), sequence_length=seq_len)
            output_LSTM_lstm = tf.reshape(tf.concat(1, outputs_lstm), [-1, hidden_size])
            output_lstm = output_LSTM_lstm

        softmax_w_lstm = tf.get_variable("softmax_w_lstm", [hidden_size, vocab_size], dtype=data_type())
        softmax_b_lstm = tf.get_variable("softmax_b_lstm", [vocab_size], dtype=data_type())            
            
        self._cost, self._nb_words_in_batch  = get_loss_function(output_lstm, softmax_w_lstm, softmax_b_lstm, labels, input_, is_training)

        cost_lstm = self._cost / (self._nb_words_in_batch + 1e-32)
        self._final_state = state_lstm
        
        if not is_training:
            return      
            
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._lr = tf.Variable(0.0, trainable=False)
        
        tvars_lstm = [embedding_reg, embedding_cbow, softmax_w_lstm, softmax_b_lstm] + [v for v in tf.trainable_variables() if v.name.startswith(lstm_lstm.name)] + [v for v in tf.trainable_variables() if v.name.startswith(cbow.name)]
        grads_lstm, _ = tf.clip_by_global_norm(tf.gradients(cost_lstm, tvars_lstm),config.max_grad_norm)   
        optimizer = get_optimizer(self._lr)
        self._train_op_lstm = optimizer.apply_gradients(zip(grads_lstm, tvars_lstm),global_step=self._global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
        
    @property
    def nb_words_in_batch(self):
        return self._nb_words_in_batch    

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def global_step(self):
        return self._global_step
        
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

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op_lstm


class config_cbow(object):
    init_scale = FLAGS.init_scale
    learning_rate = FLAGS.learning_rate
    max_grad_norm = FLAGS.max_grad_norm
    num_layers = FLAGS.num_layers
    hidden_size = FLAGS.hidden_size
    max_epoch = FLAGS.max_epoch
    max_max_epoch = FLAGS.max_max_epoch
    keep_prob = FLAGS.keep_prob
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_steps = FLAGS.num_steps
    embedded_size_reg = FLAGS.embedded_size_reg
    embedded_size_cbow = FLAGS.embedded_size_cbow
    num_history = FLAGS.num_history                     
    cbow_exp_decay = FLAGS.cbow_exp_decay

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

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0, pos_epoch = 0):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    processed_words = 0
    save_np = np.array([[0,0,0,0]])

    fetches = {"cost":model.cost, "nb_words_in_batch": model.nb_words_in_batch}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(pos_epoch, model.input.epoch_size):
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
        processed_words += sum(batch_seq_len)

        if verbose and step % (model.input.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 processed_words / (time.time() - start_time)))
            save_np = np.append(save_np, [[epoch_nb, step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 processed_words / (time.time() - start_time)]],axis=0)
    if not verbose:
        print("with(perplexity: %.3f) speed: %.0f wps" % (np.exp(costs / iters),
						 processed_words / (time.time() - start_time)))
    save_np = np.append(save_np,[[epoch_nb, 1,np.exp(costs / iters),0]],axis=0)		 
    return np.exp(costs/iters), save_np[1:]

 
def main(_):
    print('job started')
    train_name = 'ds.train.txt'
    valid_name = 'ds.valid.txt'
    test_name = 'ds.test.txt'

    config = config_cbow()
    
    eval_config = config_cbow()
    eval_config.batch_size = 1
    eval_config.num_steps = 79 #de langste zin moet hier in passen
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        tf.set_random_seed(1)

        with tf.name_scope("train"):
            train_data = reader.ds_data_sentence_with_history(config.batch_size, config.num_history, FLAGS.data_path, train_name)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = ds_cbow_sentence_model(is_training=True, config=config, input_=train_data)

        with tf.name_scope("valid"):
            valid_data = reader.ds_data_sentence_with_history(config.batch_size, config.num_history, FLAGS.data_path, valid_name)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = ds_cbow_sentence_model(is_training=False, config=config, input_=valid_data)

        with tf.name_scope("test"):
            test_data = reader.ds_data_sentence_with_history(eval_config.batch_size, eval_config.num_history, FLAGS.data_path, test_name)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = ds_cbow_sentence_model(is_training=False, config=eval_config, input_=test_data)
				
        param_train_np = np.array([['init_scale',config.init_scale], ['learning_rate', config.learning_rate],
                                   ['max_grad_norm', config.max_grad_norm], ['num_layers', config.num_layers], 
                                   ['num_history', config.num_history], ['hidden_size', config.hidden_size], 
                                   ['embedded_size_reg', config.embedded_size_reg],['embedded_size_cbow', config.embedded_size_cbow],
                                   ['max_epoch', config.max_epoch], ['max_max_epoch', config.max_max_epoch],
                                   ['keep_prob', config.keep_prob], ['lr_decay', config.lr_decay], ['cbow_exp_decay', config.cbow_exp_decay],
                                   ['batch_size', config.batch_size], ['vocab_size', train_data.pad_id], ['num_steps', config.num_steps], 
                                   ['optimizer', FLAGS.optimizer], ['loss_function', FLAGS.loss_function],  
                                   ['cbow_position', FLAGS.position],  ['cbow_combination', FLAGS.combination],
                                   ['test_name',FLAGS.test_name + str(FLAGS.num_run)]])
        
        if (os.path.exists((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results_temp' +'.npz'))):
            a = np.load((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results_temp' +'.npz'))
            train_np = a["train_np"]
            valid_np = a['valid_np']
        else:
            train_np = np.array([[0,0,0,0]])
            valid_np = np.array([[0,0,0,0]])
            
        #conf = tf.ConfigProto()
        #conf.gpu_options.allow_growth=True

        sv = tf.train.Supervisor(summary_writer=None,save_model_secs=300, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session() as session:
            start_epoch = session.run(m.global_step) // m.input.epoch_size
            pos_epoch = session.run(m.global_step) % m.input.epoch_size
            m.input.assign_batch_id(pos_epoch) 
        
            for i in range(start_epoch, config.max_max_epoch):
                if sv.should_stop():
                    break
                    
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
    				
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
    				
                train_perplexity, tra_np = run_epoch(session, m, eval_op=m.train_op, verbose=True, epoch_nb=i, pos_epoch = pos_epoch)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                pos_epoch = 0
    				
                valid_perplexity, val_np = run_epoch(session, mvalid, epoch_nb = i)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                    
                train_np = np.append(train_np, tra_np, axis=0)
                valid_np= np.append(valid_np, val_np, axis=0)
                np.savez((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results_temp' +'.npz'), train_np = train_np, valid_np=valid_np)
                    		
                #early stopping
                early_stopping = 3; #new valid_PPL will be compared to the previous 3 valid_PPL: if it is bigger than the maximun of the 3 previous, it will stop
                if i>early_stopping-1:
                    if valid_np[i+1][2] > np.max(valid_np[i+1-early_stopping:i],axis=0)[2]:
                        break
                
            test_perplexity, test_np = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
            
            if FLAGS.save_path:
                print("Saving model to %s." % (FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)  + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)))
                sv.saver.save(session, FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run) + '/' + FLAGS.test_name + '_'  + str(FLAGS.num_run), global_step=sv.global_step)
                np.savez((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results' +'.npz'), param_train_np = param_train_np, train_np = train_np[1:], valid_np=valid_np[1:], test_np = test_np)

if __name__ == "__main__":
    tf.app.run()
