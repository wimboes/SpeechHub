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

if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = '/home/wim/cuda-8.0/lib64'
        try:
            	os.system('python ' + ' '.join(sys.argv))
                sys.exit(0)
        except Exception, exc:
                print('Failed re_exec:', exc)
                sys.exit(1)

#if 'LD_LIBRARY_PATH' not in os.environ:
#        os.environ['LD_LIBRARY_PATH'] = '/users/spraak/jpeleman/tf/lib/python2.7/site-packages:/users/spraak/jpeleman/tf/cuda/lib64:/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/targets/x86_64-linux/lib'
#        try:
#            	os.system('/users/start2014/r0385169/bin/python ' + ' '.join(sys.argv))
#                sys.exit(0)
#        except Exception, exc:
#                print('Failed re_exec:', exc)
#                sys.exit(1)

import tensorflow as tf
import reader_cbow_sentence

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
#flags.DEFINE_integer("num_steps", 35, "num_steps")
flags.DEFINE_integer("num_history", 20, "num_history")
flags.DEFINE_integer("hidden_size", 512, "hidden_size")
flags.DEFINE_integer("max_epoch", 6, "max_epoch")
flags.DEFINE_integer("max_max_epoch", 39, "max_max_epoch")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob")
flags.DEFINE_float("lr_decay", 0.8, "lr_decay")
flags.DEFINE_integer("batch_size", 20, "batch_size")
flags.DEFINE_integer("vocab_size", 10000, "vocab_size")
flags.DEFINE_integer("embedded_size", 256, "embedded_size")
flags.DEFINE_integer("num_run", 0, "num_run")
flags.DEFINE_string("test_name","test","test_name")
flags.DEFINE_string("optimizer","Adagrad","optimizer")
flags.DEFINE_string("loss_function","sampled_softmax","loss_function")
flags.DEFINE_string("combination","mean","combination")
flags.DEFINE_string("position","BeforeSoftmaxCbow","position")
flags.DEFINE_string("text_data","PTB","text_data")



flags.DEFINE_string("data_path", input_path, "data_path")
flags.DEFINE_string("save_path", output_path, "save_path")
flags.DEFINE_bool("use_fp16", False, "train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
    """The input data."""
    def __init__(self, config, num_steps, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_history = num_history = config.num_history
        self.num_steps = num_steps
        self.epoch_size = (len(data) // batch_size)
        self.input_data, self.targets, self.num_steps_batch, self.history = reader_cbow_sentence.ptb_producer(data, batch_size, num_steps, num_history, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps_batch
        hidden_size = config.hidden_size
        num_history = input_.num_history
        unk_id = config.unk_id
        vocab_size = FLAGS.vocab_size + 2 #for extra 0 symbol and <bos>
        embedded_size = config.embedded_size
        history_throw_away = 100

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())
        seq_len = self.length_of_seq(input_.input_data)
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embedded_size], dtype=data_type())
            embedding2 = tf.get_variable("embedding2", [vocab_size, embedded_size], dtype=data_type())
            inputs_reg = tf.nn.embedding_lookup(embedding, input_.input_data)
            inputs_cbow = tf.nn.embedding_lookup(embedding2, input_.history)

        if is_training and config.keep_prob < 1:
            inputs_reg = tf.nn.dropout(inputs_reg, config.keep_prob)
            inputs_cbow = tf.nn.dropout(inputs_cbow, config.keep_prob)
            
        outputs_cbow = []
        for i in range(input_.num_steps):
            slice1 = tf.slice(input_.history,[0,i],[batch_size,num_history])
            slice2 = tf.slice(inputs_cbow,[0,i,0],[batch_size,num_history,embedded_size])
            
            if FLAGS.combination == "mean":
                mask = tf.cast(tf.greater(slice1,[history_throw_away]), dtype = tf.float32)
                mask1 = tf.pack([mask]*embedded_size,axis = 2)
                out = mask1*slice2
                comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1,1) + 1e-32)

            if FLAGS.combination == "exp":
                exp_weights = tf.reverse(tf.constant([[embedded_size*[np.exp(-5*k/num_history)] for k in range(num_history)] for j in range(batch_size)]),[False,True,False])
                mask = tf.cast(tf.greater(slice1,[history_throw_away]), dtype = tf.float32)
                mask1 = tf.pack([mask]*embedded_size,axis = 2)
                out = mask1*slice2*exp_weights
                comb_ = tf.reduce_sum(out,1)/(tf.reduce_sum(mask1*exp_weights,1) + 1e-32)

            outputs_cbow.append(comb_)

        
        self._temp1 = input_.input_data
        self._temp2 = input_.targets
        self._temp3 = input_.history
        self._temp4 = outputs_cbow
        self._temp5 = tf.reshape(tf.concat(1, outputs_cbow), [-1, embedded_size])
        
        if FLAGS.position == 'NoCbow':
            outputs, state = tf.nn.dynamic_rnn(cell, inputs_reg, initial_state=self._initial_state, dtype=tf.float32, sequence_length=seq_len)
            output_LSTM = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

            
            softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            output = output_LSTM

            
        if FLAGS.position == 'BeforeSoftmaxCbow':
            output_cbow = tf.reshape(tf.concat(1, outputs_cbow), [-1, embedded_size])
            
            outputs, state = tf.nn.dynamic_rnn(cell, inputs_reg, initial_state=self._initial_state, dtype=tf.float32, sequence_length=seq_len)
            output_LSTM = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
            
            softmax_w = tf.get_variable("softmax_w", [hidden_size+embedded_size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
            output = tf.concat(1,[output_LSTM,output_cbow])
            
#        if FLAGS.position == 'BeforeLSTMOnlyCbow':
#            output_cbow = tf.reshape(tf.concat(1, outputs_cbow), [-1, embedded_size])
#            
#            inputs = tf.reshape(tf.concat(1, outputs_cbow), [batch_size,tf.squeeze(num_steps), embedded_size])
#            
#            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, dtype=tf.float32)
#            output_LSTM = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
#            
#            softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
#            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
#            output = output_LSTM
#        
#        if FLAGS.position == 'BeforeLSTMCbowReg':
#            output_cbow = tf.reshape(tf.concat(1, outputs_cbow), [batch_size,tf.squeeze(num_steps), embedded_size])
#            
#            inputs = tf.concat(2,[inputs_reg, output_cbow])
#            
#            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state, dtype=tf.float32)
#            output_LSTM = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
#            
#            softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
#            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
#            output = output_LSTM

            
        loss = get_loss_function(output, softmax_w, softmax_b, input_.targets, batch_size, is_training, unk_id)
        
        self._cost = cost = loss

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
    
    def length_of_seq(self,sequence):
        used = tf.sign(tf.abs(sequence))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state
        
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
#    num_steps = FLAGS.num_steps
    hidden_size = FLAGS.hidden_size
    max_epoch = FLAGS.max_epoch
    max_max_epoch = FLAGS.max_max_epoch
    keep_prob = FLAGS.keep_prob
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    embedded_size = FLAGS.embedded_size
    num_history = FLAGS.num_history
    unk_id = 0                      #default not used word_id

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

def get_loss_function(output, softmax_w, softmax_b, targets, batch_size, is_training,unk_id):
    
    #masking of 0 id's (always) and unk_id (only during testing)
    targets = tf.reshape(targets, [-1])
    mask = tf.logical_and(tf.not_equal(targets,[0]),tf.not_equal(targets,[unk_id]))
    mask2 = tf.reshape(tf.where(mask),[-1])
    targets = tf.gather(targets, mask2)
    output = tf.gather(output, mask2)
    nb_words_in_batch = tf.reduce_sum(tf.cast(mask,dtype=tf.float32)) + 1e-32
    if FLAGS.loss_function == "full_softmax":    
        logits = tf.matmul(output, softmax_w)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
        return tf.reduce_sum(loss) / nb_words_in_batch

    if FLAGS.loss_function == 'sampled_softmax':
        if is_training:
            loss = tf.nn.sampled_softmax_loss(tf.transpose(softmax_w), softmax_b, output, tf.reshape(targets, [-1,1]), 32, FLAGS.vocab_size)
            return tf.reduce_sum(loss) / nb_words_in_batch
        else:
            logits = tf.matmul(output, softmax_w)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets, name=None)
            return tf.reduce_sum(loss) / nb_words_in_batch

    return 0

def run_epoch(session, model, eval_op=None, verbose=False, epoch_nb = 0):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    save_np = np.array([[0,0,0,0]])

    fetches = {"cost": model.cost, 'temp1': model.temp1,'temp2':model.temp2, 'temp3':model.temp3, 'temp4': model.temp4, 'temp5': model.temp5}
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        vals = session.run(fetches)
        cost = vals["cost"]
#        out = vals['temp1']
#        print(out)
#        print('\n')
#        print(vals['temp2'])
#        print('\n')
#        print(vals['temp3'])
#        print('\n')
#        print(vals['temp4'])
#        print('\n')
#        print(vals['temp5'])
        costs += cost
        iters += 1

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
    
    raw_data = reader_cbow_sentence.ptb_raw_data(FLAGS.data_path, FLAGS.text_data, FLAGS.vocab_size)
    train_data, valid_data, test_data, _, unk_id, num_steps = raw_data 
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    #eval_config.num_steps = 1
    eval_config.unk_id = unk_id
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        tf.set_random_seed(1)

        with tf.name_scope("Train"):
            train_input = PTBInput(config=config, num_steps=num_steps, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, input_=train_input)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(config=config, num_steps=num_steps, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
            tf.scalar_summary("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config, num_steps=num_steps, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)
				
        param_train_np = np.array([['init_scale',config.init_scale], ['learning_rate', config.learning_rate],
                                   ['max_grad_norm', config.max_grad_norm], ['num_layers', config.num_layers], ['num_history', config.num_history],
                                   ['num_steps', num_steps], ['hidden_size', config.hidden_size], 
                                   ['embedded_size', config.embedded_size],['max_epoch', config.max_epoch],
                                   ['max_max_epoch', config.max_max_epoch],['keep_prob', config.keep_prob], 
                                   ['lr_decay', config.lr_decay], ['batch_size', config.batch_size], ['data', FLAGS.text_data],
                                   ['vocab_size', FLAGS.vocab_size], ['optimizer', FLAGS.optimizer], 
                                   ['loss_function', FLAGS.loss_function],  ['cbow_position', FLAGS.position],  ['cbow_combination', FLAGS.combination]])
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
