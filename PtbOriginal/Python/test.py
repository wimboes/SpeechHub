import tensorflow as tf

optimizer = tf.train.GradientDescentOptimizer(1)
execfile('ptb_word_lm.py')

# optimizer = tf.train.AdadeltaOptimizer()
# execfile('ptb_word_lm.py')
# 
# optimizer = tf.train.AdagradOptimizer(1)
# execfile('ptb_word_lm.py')
# 
# optimizer = tf.train.AdamOptimizer()
# execfile('ptb_word_lm.py')

