import tensorflow as tf

cell = tf.nn.rnn_cell.BasicLSTMCell(512)
input_data = tf.constant([[1,2],[3,4]],dtype=tf.float16)
state = cell.zero_state(2, tf.float16)

with tf.variable_scope("LSTM") as vs:
    output, state = cell(input_data, state)
        
lstm_variables = [v for v in tf.all_variables()
                    if v.name.startswith(vs.name)]

init = tf.initialize_variables(lstm_variables)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, "test.ckpt")
    print("model saved")

