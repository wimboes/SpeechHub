### authors : Wim Boes & Robbe Van Rompaey
### date: 4-10-2016

import tensorflow as tf

"""

- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_unrolls - the number of unrolled steps of LSTM
- hidden_size - the size of the LSTM state
- init_epochs - the number of epochs trained with the initial learning rate
- max_epochs - the max number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "init_epochs"
- batch_size - the batch size
- vocab_size - the size of the vocabulary
- embeddded_size - the dimension of the embeddings serving as input for the LSTM cell

"""

init_scale = 0.1
learning_rate = 1.0
max_grad_norm = 1
num_layers = 1
num_unrolls = 2
hidden_size = 2
init_epochs = 1
max_epochs = 2
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
vocab_size = 10000
embedded_size = 2
data_type = tf.float32