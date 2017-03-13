from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

input_data = tf.constant([[4, 3], [5, 6], [1, 0]])
embedding = tf.get_variable("embedding", [7, 5], dtype=tf.float32)
inputs = tf.nn.embedding_lookup(embedding, input_data)

initializer = tf.initialize_all_variables()
sess = tf.Session()
sess.run(initializer)
out0 = sess.run(input_data)
out1 = sess.run(embedding)
out2 = sess.run(inputs)
