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
        print('hihi')
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
        try:
            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)


import tensorflow as tf
import reader_topic_ds

##### paths

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'input')
output_path = os.path.join(general_path,'output/output_topic_ds')

##### flags

flags = tf.flags
logging = tf.logging

### regular

flags.DEFINE_float("init_scale_reg", 0.05, "init_scale_reg")
flags.DEFINE_float("learning_rate_reg", 1, "learning_rate_reg")
flags.DEFINE_float("max_grad_norm_reg", 5, "max_grad_norm_reg")
flags.DEFINE_integer("num_layers_reg", 1, "num_layers_reg")
flags.DEFINE_integer("hidden_size_reg", 256, "hidden_size_reg")
flags.DEFINE_integer("max_epoch_reg", 5, "max_epoch_reg")
flags.DEFINE_integer("max_max_epoch_reg", 10, "max_max_epoch_reg")
flags.DEFINE_float("keep_prob_reg", 0.5, "keep_prob_reg")
flags.DEFINE_float("lr_decay_reg", 1, "lr_decay_reg")
flags.DEFINE_integer("embedded_size_reg", 128, "embedded_size_reg")

### lda

flags.DEFINE_float("init_scale_lda", 0.05, "init_scale_lda")
flags.DEFINE_float("learning_rate_lda", 1, "learning_rate_lda")
flags.DEFINE_float("max_grad_norm_lda", 5, "max_grad_norm_lda")
flags.DEFINE_integer("num_layers_lda", 1, "num_layers_lda")
flags.DEFINE_integer("hidden_size_lda", 256, "hidden_size_lda")
flags.DEFINE_integer("max_epoch_lda", 5, "max_epoch_lda")
flags.DEFINE_integer("max_max_epoch_lda", 10, "max_max_epoch_lda")
flags.DEFINE_float("keep_prob_lda", 0.5, "keep_prob_lda")
flags.DEFINE_float("lr_decay_lda", 0.8, "lr_decay_lda")
flags.DEFINE_integer("embedded_size_lda", 128, "embedded_size_lda")

### general

flags.DEFINE_string("mode", "reg", "mode")
flags.DEFINE_integer("batch_size", 40, "batch_size")
flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","topic","test_name")
flags.DEFINE_string("data_path",input_path,"data_path")
flags.DEFINE_string("save_path",output_path,"save_path")
flags.DEFINE_string("use_fp16",False,"train blabla")
flags.DEFINE_string("loss_function","full_softmax","loss_function")
flags.DEFINE_string("optimizer","Adagrad","optimizer")

FLAGS = flags.FLAGS

##### classes and functions 

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class ds_input(object):
    def __init__(self, config, max_length, length_array, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.average_sentence_length = np.mean(length_array)
        self.epoch_size = (len(data) // batch_size)
        self.input_data, self.targets, self.sentence_lengths_batch = reader_topic_ds.ds_producer(data, batch_size, length_array, max_length, name=None)

class ds_topic_model(object):
    def __init__(self, is_training, config, input_, topic_matrix, initializer_reg, initializer_lda):
        self._input = input_

        batch_size = input_.batch_size
        vocab_size = config.vocab_size 
        nb_topics = topic_matrix.get_shape()[0]

        seq_len = self.length_of_seq(input_.input_data, vocab_size)
        
        with tf.device("/cpu:0"):
            embedding_reg = tf.get_variable("embedding_reg", [vocab_size+1, config.embedded_size_reg], dtype=data_type(), initializer = initializer_reg)
            inputs_reg = tf.nn.embedding_lookup(embedding_reg, input_.input_data)
            embedding_lda = tf.get_variable("embedding_lda", [vocab_size+1, config.embedded_size_lda], dtype=data_type(), initializer = initializer_lda)
            inputs_lda = tf.nn.embedding_lookup(embedding_lda, input_.input_data)
            
        if is_training and config.keep_prob_reg < 1:
            inputs_reg = tf.nn.dropout(inputs_reg, config.keep_prob_reg)
        
        if is_training and config.keep_prob_lda < 1:
            inputs_lda = tf.nn.dropout(inputs_lda, config.keep_prob_lda)

        with tf.variable_scope('reg_lstm', initializer = initializer_reg) as reg_lstm:
            lstm_cell_reg = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size_reg, forget_bias=0.0, state_is_tuple=True)
            if is_training and config.keep_prob_reg < 1:
                lstm_cell_reg = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_reg, output_keep_prob=config.keep_prob_reg)
            cell_reg = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_reg] * config.num_layers_reg, state_is_tuple=True)

            self._initial_state_reg = cell_reg.zero_state(batch_size, data_type())
            
            outputs_reg, state_reg = tf.nn.dynamic_rnn(cell_reg, inputs_reg, initial_state=self._initial_state_reg, dtype=tf.float32, sequence_length=seq_len)
            output_reg = tf.reshape(tf.concat(1, outputs_reg), [-1, config.hidden_size_reg])
            
        with tf.variable_scope('lda_lstm', initializer = initializer_lda) as lda_lstm:
            lstm_cell_lda = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size_lda, forget_bias=0.0, state_is_tuple=True)
            if is_training and config.keep_prob_lda < 1:
                lstm_cell_lda = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_lda, output_keep_prob=config.keep_prob_lda)
            cell_lda = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_lda] * config.num_layers_lda, state_is_tuple=True)

            self._initial_state_lda = cell_lda.zero_state(batch_size, data_type())
            
            outputs_lda, state_lda = tf.nn.dynamic_rnn(cell_lda, inputs_lda, initial_state=self._initial_state_lda, dtype=tf.float32, sequence_length=seq_len)
            output_lda = tf.reshape(tf.concat(1, outputs_lda), [-1, config.hidden_size_lda])

        softmax_w_reg = tf.get_variable("softmax_w_reg", [config.hidden_size_reg, vocab_size], dtype=data_type(), initializer = initializer_reg)
        softmax_b_reg = tf.get_variable("softmax_b_reg", [vocab_size], dtype=data_type(), initializer = initializer_reg)
        
        softmax_w_lda = tf.get_variable("softmax_w_lda", [config.hidden_size_lda, nb_topics], dtype=data_type(), initializer = initializer_lda)
        softmax_b_lda = tf.get_variable("softmax_b_lda", [nb_topics], dtype=data_type(), initializer = initializer_lda)
        
        
        self._interpol = tf.Variable(0.0)
        self._new_interpol = tf.placeholder(tf.float32, shape=[], name="new_interpol")
        self._interpol_update = tf.assign(self._interpol, self._new_interpol)
    
        loss = get_loss_function(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, self._interpol, input_.targets, topic_matrix, batch_size, is_training, vocab_size, config.unk_id)

        self._cost = cost = loss
        
        self._final_state_reg = state_reg
        self._final_state_lda = state_lda
        #self._temp1 = self._interpol
        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)

    
        tvars_reg = [embedding_reg, softmax_w_reg, softmax_b_reg] + [v for v in tf.trainable_variables() if v.name.startswith(reg_lstm.name)]
        grads_reg, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars_reg),config.max_grad_norm_reg)    
        tvars_lda = [embedding_lda, softmax_w_lda, softmax_b_lda] + [v for v in tf.trainable_variables() if v.name.startswith(lda_lstm.name)]
        grads_lda, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars_lda),config.max_grad_norm_lda)  
        tvars_int = [self._interpol]
        grads_int, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars_int),5)

        optimizer = get_optimizer(self._lr)
        
        self._train_op_reg = optimizer.apply_gradients(zip(grads_reg, tvars_reg),global_step=tf.contrib.framework.get_or_create_global_step())
        self._train_op_lda = optimizer.apply_gradients(zip(grads_lda, tvars_lda),global_step=tf.contrib.framework.get_or_create_global_step())
        self._train_op_int = optimizer.apply_gradients(zip(grads_int, tvars_int),global_step=tf.contrib.framework.get_or_create_global_step())
        
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def length_of_seq(self,sequence, vocab_size):
        used = tf.sign(tf.abs(sequence-vocab_size))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
        
    def assign_interpol(self, session, interpol_value):
        session.run(self._interpol_update, feed_dict={self._new_interpol: interpol_value})

    @property
    def temp1(self):
        return self._temp1
    @property
    def input(self):
        return self._input

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

    @property
    def lr(self):
        return self._lr
        
    @property
    def train_op_reg(self):
        return self._train_op_reg
        
    @property
    def train_op_lda(self):
        return self._train_op_lda
        
    @property
    def train_op_int(self):
        return self._train_op_int
        


class config_topic(object):
    init_scale_reg = FLAGS.init_scale_reg
    learning_rate_reg = FLAGS.learning_rate_reg
    max_grad_norm_reg = FLAGS.max_grad_norm_reg
    num_layers_reg = FLAGS.num_layers_reg
    hidden_size_reg = FLAGS.hidden_size_reg
    max_epoch_reg = FLAGS.max_epoch_reg
    max_max_epoch_reg = FLAGS.max_max_epoch_reg
    keep_prob_reg = FLAGS.keep_prob_reg
    lr_decay_reg = FLAGS.lr_decay_reg
    embedded_size_reg = FLAGS.embedded_size_reg

    init_scale_lda = FLAGS.init_scale_lda
    learning_rate_lda = FLAGS.learning_rate_lda
    max_grad_norm_lda = FLAGS.max_grad_norm_lda
    num_layers_lda = FLAGS.num_layers_lda
    hidden_size_lda = FLAGS.hidden_size_lda
    max_epoch_lda = FLAGS.max_epoch_lda
    max_max_epoch_lda = FLAGS.max_max_epoch_lda
    keep_prob_lda = FLAGS.keep_prob_lda
    lr_decay_lda = FLAGS.lr_decay_lda
    embedded_size_lda = FLAGS.embedded_size_lda

    batch_size = FLAGS.batch_size    
    vocab_size = 0
    mode = FLAGS.mode
    unk_id = 0
    
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

def get_loss_function(output_reg, output_lda, softmax_w_reg, softmax_w_lda, softmax_b_reg, softmax_b_lda, interpol, targets, topic_matrix, batch_size, is_training, vocab_size, unk_id):
    if FLAGS.loss_function == "full_softmax":
        targetsz = tf.reshape(targets, [-1])
        maskz = tf.logical_and(tf.not_equal(targetsz,[vocab_size]),tf.not_equal(targetsz,[unk_id]))
        mask2z  = tf.reshape(tf.where(maskz),[-1])
        targetsz = tf.gather(targetsz, mask2z)
        output_regz = tf.gather(output_reg, mask2z) #?
        output_ldaz = tf.gather(output_lda, mask2z) #?
        nb_words_in_batchz = tf.reduce_sum(tf.cast(maskz,dtype=tf.float32)) + 1e-32
        
        logits_reg = tf.matmul(output_regz, softmax_w_reg) + softmax_b_reg
        probs_reg = tf.nn.softmax(logits_reg) 
        
        logits_lda = tf.matmul(output_ldaz, softmax_w_lda) + softmax_b_lda
        probs_topic = tf.nn.softmax(logits_lda) 
        probs_lda = tf.matmul(probs_topic,topic_matrix)
        
        probs = (1-interpol)*probs_reg + (interpol)*probs_lda
                
        idx = tf.reshape(targetsz, [-1])
        idx_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1] + idx
        y = tf.gather(tf.reshape(probs, [-1]), idx_flattened)  # use flattened indices
        loss = -tf.log(y)
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(targets, [-1]), name=None)
        return tf.reduce_sum(loss) / nb_words_in_batchz

        #    if FLAGS.loss_function == 'sampled_softmax':
#        if is_training:
#            loss = tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1, 1]), 32, FLAGS.vocab_size)
#            return tf.reduce_sum(loss) / batch_size
#        else:
#            logits = tf.matmul(output, softmax_w) + softmax_b + topic_matrix
#            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(targets, [-1]), name=None)
#            return tf.reduce_sum(loss) / batch_size
#    if FLAGS.loss_function == 'noise_contrastive_estimation':
#        if is_training:
#            loss = tf.nn.nce_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1, 1]), 32, FLAGS.vocab_size)
#            return tf.reduce_sum(loss) / batch_size
#        else:
#            logits = tf.matmul(output, softmax_w) + softmax_b
#            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(targets, [-1]), name=None)
#            return tf.reduce_sum(loss) / batch_size
#    return 0

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state_lda)
    save_np = np.array([[0,0,0,0]])

    fetches = {"cost": model.cost,"final_state_lda": model.final_state_lda}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state_lda):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state_lda"]
#        print(vals['temp1'])

        costs += cost
        iters += 1

        if verbose and step % (model.input.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 iters * model.input.batch_size * model.input.average_sentence_length / (time.time() - start_time)))
            save_np = np.append(save_np, [[epoch_nb, step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 iters * model.input.batch_size * model.input.average_sentence_length / (time.time() - start_time)]],axis=0)
    save_np = np.append(save_np,[[epoch_nb, 1,np.exp(costs / iters),0]],axis=0)		 
    return np.exp(costs/iters), save_np[1:]

 
def main(_):
    print('job started')
    
    raw_data =  reader_topic_ds.ds_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, vocab_size, unk_id, max_length, topic_array, train_length_array, valid_length_array, test_length_array = raw_data 
    nb_topics = np.shape(topic_array)[0]
    
    config = config_topic()
    config.vocab_size = vocab_size
    config.unk_id = vocab_size + 1
    eval_config = config_topic()
    eval_config.batch_size = 1
    eval_config.vocab_size = vocab_size
    eval_config.unk_id = unk_id
    
    with tf.Graph().as_default():    
        
        initializer_reg = tf.random_uniform_initializer(-config.init_scale_reg, config.init_scale_reg)
        initializer_lda = tf.random_uniform_initializer(-config.init_scale_lda, config.init_scale_lda)
        tf.set_random_seed(1)
        
        topic_matrix = tf.constant(topic_array,dtype=tf.float32)

        with tf.name_scope("train"):
            train_input = ds_input(config=config, max_length=max_length, length_array=train_length_array, data=train_data, name="train_input")
            with tf.variable_scope("model", reuse=None):
            	m = ds_topic_model(is_training=True, config=config, input_ = train_input, topic_matrix = topic_matrix, initializer_reg = initializer_reg, initializer_lda = initializer_lda)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        with tf.name_scope("valid"):
            valid_input = ds_input(config=config, max_length=max_length, length_array=valid_length_array, data=valid_data, name="valid_input")
            with tf.variable_scope("model", reuse=True):
                mvalid = ds_topic_model(is_training=False, config=config, input_ = valid_input, topic_matrix = topic_matrix, initializer_reg = initializer_reg, initializer_lda = initializer_lda)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("test"):
            test_input = ds_input(config=eval_config, max_length=max_length, length_array=test_length_array, data=test_data, name="test_input")
            with tf.variable_scope("model", reuse=True):
                mtest = ds_topic_model(is_training=False, config=eval_config, input_ = test_input, topic_matrix = topic_matrix, initializer_reg = initializer_reg, initializer_lda = initializer_lda) 

        param_train_np = np.array([['init_scale_reg',config.init_scale_reg], ['init_scale_lda',config.init_scale_lda],
                                   ['learning_rate_reg', config.learning_rate_reg], ['learning_rate_lda', config.learning_rate_lda],
                                   ['max_grad_norm_reg', config.max_grad_norm_reg], ['max_grad_norm_lda', config.max_grad_norm_lda],
                                   ['num_layers_reg', config.num_layers_reg], ['num_layers_lda', config.num_layers_lda],
                                   ['hidden_size_reg', config.hidden_size_reg], ['hidden_size_lda', config.hidden_size_lda],
                                   ['embedded_size_reg', config.embedded_size_reg], ['embedded_size_lda', config.embedded_size_lda],
                                   ['max_epoch_reg', config.max_epoch_reg], ['max_epoch_lda', config.max_epoch_lda],
                                   ['max_max_epoch_reg', config.max_max_epoch_reg], ['max_max_epoch_lda', config.max_max_epoch_lda],
                                   ['keep_prob_reg', config.keep_prob_reg], ['keep_prob_lda', config.keep_prob_lda],
                                   ['nb_topics', nb_topics], ['vocab_size', config.vocab_size], ['batch_size', config.batch_size],
                                   ['lr_decay_reg', config.lr_decay_reg], ['lr_decay_lda', config.lr_decay_lda],
                                   ['optimizer', FLAGS.optimizer],
                                   ['loss_function', FLAGS.loss_function], 
                                   ['mode', FLAGS.mode], ['test_name',FLAGS.test_name + str(FLAGS.num_run)]])
        
        train_np = np.array([[0,0,0,0]])
        valid_np = np.array([[0,0,0,0]])

        
        sv = tf.train.Supervisor(summary_writer=None,logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session() as session:
            if FLAGS.mode == 'reg':
                m.assign_interpol(session, 0.0)
                mvalid.assign_interpol(session, 0.0)
                mtest.assign_interpol(session, 0.0)
                for i in range(config.max_max_epoch_reg):
                    lr_decay = config.lr_decay_reg ** max(i - config.max_epoch_reg, 0.0)
                    m.assign_lr(session, config.learning_rate_reg * lr_decay)
                     
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				
                    train_perplexity, tra_np = run_epoch(session, m, eval_op=m.train_op_reg, verbose=True, epoch_nb=i)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				
                    valid_perplexity, val_np = run_epoch(session, mvalid, epoch_nb = i)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
				
                    train_np = np.append(train_np, tra_np, axis=0)
                    valid_np= np.append(valid_np, val_np, axis=0)
                		
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
                    np.savez((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/resultsreg' +'.npz'), param_train_np = param_train_np, train_np = train_np[1:], valid_np=valid_np[1:], test_np = test_np)

            elif FLAGS.mode == 'lda':
                m.assign_interpol(session, 1.0)
                mvalid.assign_interpol(session, 1.0)
                mtest.assign_interpol(session, 1.0)
                for i in range(config.max_max_epoch_lda):
                    lr_decay = config.lr_decay_lda ** max(i - config.max_epoch_lda, 0.0)
                    m.assign_lr(session, config.learning_rate_lda * lr_decay)
                     
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				
                    train_perplexity, tra_np = run_epoch(session, m, eval_op=m.train_op_lda, verbose=True, epoch_nb=i)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				
                    valid_perplexity, val_np = run_epoch(session, mvalid, epoch_nb = i)
                    print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
				
                    train_np = np.append(train_np, tra_np, axis=0)
                    valid_np= np.append(valid_np, val_np, axis=0)
                		
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
                    
            elif FLAGS.mode == 'int':
                interpol_values = np.linspace(0,1,num=50)
                interpol_best = 0
                interpol_best_perplexity = 1e9
                for interpol_value in interpol_values:
                    mvalid.assign_interpol(session, interpol_value)                
                    valid_perplexity, _ = run_epoch(session, mvalid)
                    print("Interpolation factor = %.3f : valid perplexity = %.3f" % (interpol_value, valid_perplexity))
                    if valid_perplexity < interpol_best_perplexity:
                        interpol_best = interpol_value
                        interpol_best_perplexity = valid_perplexity
                print("Best interpolation factor = %.3f" % interpol_best)
                m.assign_interpol(session, interpol_best)  
                mtest.assign_interpol(session, interpol_best) 
                test_perplexity, test_np = run_epoch(session, mtest)
                print("Test Perplexity: %.3f" % test_perplexity)
                if FLAGS.save_path:
                    print("Saving model to %s." % (FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)  + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)))
                    sv.saver.save(session, FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run) + '/' + FLAGS.test_name + '_'  + str(FLAGS.num_run), global_step=sv.global_step)
                    np.savez((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/resultsint' +'.npz'), param_train_np = param_train_np, train_np = train_np[1:], valid_np=valid_np[1:], test_np = test_np)
                    
                    
if __name__ == "__main__":
    tf.app.run()
