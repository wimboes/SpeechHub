# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np

#if 'LD_LIBRARY_PATH' not in os.environ:
#        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/users/start2014/r0385169/.local/cudnn'
#        try:
#            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
#                sys.exit(0)
#        except Exception, exc:
#                print('Failed re_exec:', exc)
#                sys.exit(1)



import tensorflow as tf
import reader

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'input')
output_path = os.path.join(general_path,'output/output_original_ds')

# set data and save path

flags = tf.flags
logging = tf.logging

flags.DEFINE_float("init_scale", 0.05, "init_scale")
flags.DEFINE_float("learning_rate", 1, "learning_rate")
flags.DEFINE_float("max_grad_norm", 5, "max_grad_norm")
flags.DEFINE_integer("num_layers", 1, "num_layers")
flags.DEFINE_integer("num_steps", 50, "num_steps")
flags.DEFINE_integer("hidden_size", 128, "hidden_size")
flags.DEFINE_integer("max_epoch", 3, "max_epoch")
flags.DEFINE_integer("max_max_epoch", 5, "max_max_epoch")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
flags.DEFINE_float("lr_decay", 0.8, "lr_decay")
flags.DEFINE_integer("batch_size", 5, "batch_size")
flags.DEFINE_integer("embedded_size", 64, "embedded_size")
flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","original","test_name")
flags.DEFINE_string("optimizer","Adagrad","optimizer")
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
        self._num_steps = num_steps = config.num_steps
        hidden_size = config.hidden_size
        vocab_size = input_.pad_id 
        embedded_size = config.embedded_size
        
        self._data = data =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_data')
        self._labels = labels =  tf.placeholder(tf.int32, [batch_size, num_steps], name = 'batch_labels')
        self._seq_len = seq_len =  tf.placeholder(tf.int32, [batch_size], name = 'seq_len')

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size + 1, embedded_size], dtype=data_type()) #om pad symbool toe te laten
            inputs = tf.nn.embedding_lookup(embedding, data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, dtype=data_type(), sequence_length = seq_len)
        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        loss = get_loss_function(output, softmax_w, softmax_b, labels, input_, is_training)
        
        self._cost = cost = loss
        self._final_state = state

        if not is_training:
            return
        
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        optimizer = get_optimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self._global_step)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input
        
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

    @property
    def lr(self):
        return self._lr
        
    @property
    def global_step(self):
        return self._global_step

    @property
    def train_op(self):
        return self._train_op


class config_original(object):
    init_scale = FLAGS.init_scale
    learning_rate = FLAGS.learning_rate
    max_grad_norm = FLAGS.max_grad_norm
    num_layers = FLAGS.num_layers
    num_steps = FLAGS.num_steps
    hidden_size = FLAGS.hidden_size
    max_epoch = FLAGS.max_epoch
    max_max_epoch = FLAGS.max_max_epoch
    keep_prob = FLAGS.keep_prob
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    embedded_size = FLAGS.embedded_size

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
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32)) + 1e-32

    if FLAGS.loss_function == "full_softmax":
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
        return tf.reduce_sum(loss) / nb_words_in_batch

    if FLAGS.loss_function == 'sampled_softmax':
        if is_training:
            loss = tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1,1]), 32, data.pad_id)
            return tf.reduce_sum(loss) / nb_words_in_batch
        else:
            logits = tf.matmul(output, softmax_w) + softmax_b
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
            return tf.reduce_sum(loss) / nb_words_in_batch

    return 0

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0, pos_epoch = 0):
    start_time = time.time()
    costs = 0.0
    iters = 0
    processed_words = 0
    state = session.run(model.initial_state)
    save_np = np.array([[0,0,0,0]])

    fetches = {"cost": model.cost,"final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(pos_epoch, model.input.epoch_size):
        batch_data, batch_labels = model.input.next_batch()
        feed_dict = {}
        feed_dict[model.data] = batch_data
        feed_dict[model.labels] = batch_labels
        feed_dict[model.seq_len] = np.ones(model.input.batch_size)*model.input.num_steps
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += 1 
        processed_words += sum(np.ones(model.input.batch_size)*model.input.num_steps)

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
    train_name = 'ds.testshort.txt'
    valid_name = 'ds.testshort.txt'
    test_name = 'ds.test.txt'

    
    config = config_original()
    
    eval_config = config_original()
    eval_config.batch_size = 1
    eval_config.num_steps = 1 

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        tf.set_random_seed(1)

        with tf.name_scope("train"):
            train_data = reader.ds_data_continuous(config.batch_size, FLAGS.num_steps, FLAGS.data_path, train_name)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = ds_original_model(is_training=True, config=config, input_=train_data)

        with tf.name_scope("valid"):
            valid_data = reader.ds_data_continuous(config.batch_size, FLAGS.num_steps, FLAGS.data_path, valid_name)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = ds_original_model(is_training=False, config=config, input_=valid_data)

        with tf.name_scope("test"):
            test_data = reader.ds_data_continuous(eval_config.batch_size, eval_config.num_steps, FLAGS.data_path, test_name)
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = ds_original_model(is_training=False, config=eval_config, input_=test_data)
				
        param_train_np = np.array([['init_scale',config.init_scale], ['learning_rate', config.learning_rate],
                                   ['max_grad_norm', config.max_grad_norm], ['num_layers', config.num_layers],
                                   ['num_steps', config.num_steps], ['hidden_size', config.hidden_size], 
                                   ['embedded_size', config.embedded_size],['max_epoch', config.max_epoch],
                                   ['max_max_epoch', config.max_max_epoch],['keep_prob', config.keep_prob], 
                                   ['lr_decay', config.lr_decay], ['batch_size', config.batch_size], 
                                   ['vocab_size', train_data.pad_id], ['optimizer', FLAGS.optimizer], 
                                   ['loss_function', FLAGS.loss_function], ['test_name',FLAGS.test_name + str(FLAGS.num_run)]])
        
        if (os.path.exists((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results_temp' +'.npz'))):
            a = np.load((FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run)+ '/results_temp' +'.npz'))
            train_np = a["train_np"]
            valid_np = a['valid_np']
        else:
            train_np = np.array([[0,0,0,0]])
            valid_np = np.array([[0,0,0,0]])
		
        sv = tf.train.Supervisor(summary_writer=None,save_model_secs=60, logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
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
				
                train_perplexity, tra_np = run_epoch(session, m, eval_op=m.train_op, verbose=True, epoch_nb=i, pos_epoch=pos_epoch)
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
