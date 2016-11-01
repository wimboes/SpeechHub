# authors : Wim Boes & Robbe Van Rompaey
# date: 11-10-2016 

# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import tensorflow as tf
import reader

#if 'LD_LIBRARY_PATH' not in os.environ:
#        os.environ['LD_LIBRARY_PATH'] = '/users/spraak/jpeleman/tf/lib/python2.7/site-packages:/users/spraak/jpeleman/tf/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/targets/x86_64-linux/lib'
#        try:
#            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
#                sys.exit(0)
#        except Exception, exc:
#                print('Failed re_exec:', exc)
#                sys.exit(1)

python_path = os.path.abspath(os.getcwd())
general_path = os.path.split(python_path)[0]
input_path = os.path.join(general_path,'Input')
output_path = os.path.join(general_path,'Output')

# set data and save path

flags = tf.flags
logging = tf.logging

flags.DEFINE_float("init_scale", 0.05, "init_scale")
flags.DEFINE_float("learning_rate", 1, "learning_rate")
flags.DEFINE_float("max_grad_norm", 5, "max_grad_norm")
flags.DEFINE_integer("num_layers", 1, "num_layers")
flags.DEFINE_integer("num_steps", 10, "num_steps")
flags.DEFINE_integer("hidden_size", 256, "hidden_size")
flags.DEFINE_integer("max_epoch", 6, "max_epoch")
flags.DEFINE_integer("max_max_epoch", 32, "max_max_epoch")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
flags.DEFINE_float("lr_decay", 0.8, "lr_decay")
flags.DEFINE_integer("batch_size", 20, "batch_size")
flags.DEFINE_integer("vocab_size", 10002, "vocab_size")
flags.DEFINE_integer("embedded_size", 128, "embedded_size")
flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","askoy","test_name")
flags.DEFINE_string("optimizer","GradDesc","optimizer")
flags.DEFINE_string("loss_function","sequence_loss_by_example","loss_function")

flags.DEFINE_string("data_path", input_path, "data_path")
flags.DEFINE_string("save_path", output_path, "save_path")
flags.DEFINE_bool("use_fp16", False, "train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)



class PTBModel(object):
    """The PTB model."""
    
    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        embedded_size = config.embedded_size

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embedded_size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #					 for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        loss = get_loss_function(output, softmax_w, softmax_b,input_.targets, batch_size,num_steps, is_training)
        
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        optimizer = get_optimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

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
        return self._train_op


class Config(object):
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
    vocab_size = FLAGS.vocab_size
    embedded_size = FLAGS.embedded_size
    optimizer = FLAGS.optimizer
    loss_function = FLAGS.loss_function

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

def get_loss_function(output, softmax_w, softmax_b,targets, batch_size, num_steps, is_training):
    if FLAGS.loss_function == "sequence_loss_by_example":
        logits = tf.matmul(output, softmax_w) + softmax_b
        return tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(targets, [-1])], [tf.ones([batch_size * num_steps], dtype=data_type())])
    if FLAGS.loss_function == 'sampled_softmax':
        if is_training:
            return tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1, 1]), 512, FLAGS.vocab_size)
        else:
            logits = tf.matmul(output, softmax_w) + softmax_b
            return tf.nn.seq2seq.sequence_loss_by_example([logits],[tf.reshape(targets, [-1])], [tf.ones([batch_size * num_steps], dtype=data_type())])	
    return 0

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    save_np = np.array([[0,0,0,0]])

    fetches = {"cost": model.cost,"final_state": model.final_state}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 iters * model.input.batch_size / (time.time() - start_time)))
            save_np = np.append(save_np, [[epoch_nb, step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
						 iters * model.input.batch_size / (time.time() - start_time)]],axis=0)
    save_np = np.append(save_np,[[epoch_nb, 1,np.exp(costs / iters),0]],axis=0)		 
    return np.exp(costs/iters), save_np[1:]


def get_config():
    return Config()
 
def main(_):
    print('job started')
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data 
    
    config = get_config()
    print(FLAGS.init_scale)
    print(FLAGS.learning_rate)
    print(FLAGS.max_grad_norm)
    print(FLAGS.num_layers)
    print(FLAGS.num_steps)
    print(FLAGS.hidden_size)
    print(FLAGS.max_epoch)
    print(FLAGS.max_max_epoch)
    print(FLAGS.keep_prob)
    print(FLAGS.lr_decay)
    print(FLAGS.batch_size)
    print(FLAGS.vocab_size)
    print(FLAGS.embedded_size)
    print(FLAGS.optimizer)
    print(FLAGS.loss_function)
    print(FLAGS.num_run)
    print(FLAGS.test_name)

    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        tf.set_random_seed(1)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
				
        param_train_np = np.array([['init_scale',config.init_scale], ['learning_rate', config.learning_rate],
                                   ['max_grad_norm', config.max_grad_norm], ['num_layers', config.num_layers],
                                   ['num_steps', config.num_steps], ['hidden_size', config.hidden_size], 
                                   ['embedded_size', config.embedded_size],['max_epoch', config.max_epoch],
                                   ['max_max_epoch', config.max_max_epoch],['keep_prob', config.keep_prob], 
                                   ['lr_decay', config.lr_decay], ['batch_size', config.batch_size], 
                                   ['vocab_size', config.vocab_size], ['optimizer', config.optimizer], 
                                   ['loss_function', config.loss_function]])
        train_np = np.array([[0,0,0,0]])
        valid_np = np.array([[0,0,0,0]])
		
        sv = tf.train.Supervisor(summary_writer=None,logdir=FLAGS.save_path + '/' + FLAGS.test_name + '_' + str(FLAGS.num_run))
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
				
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				
                train_perplexity, tra_np = run_epoch(session, m, eval_op=m.train_op, verbose=True, epoch_nb=i)
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

if __name__ == "__main__":
    tf.app.run()
