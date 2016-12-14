import tensorflow as tf


weights = tf.Variable([0,0,3])
saver = tf.train.Saver()

sess =  tf.Session()
saver.restore(sess, "test.ckpt")
print("model restored")
a = sess.run(weights)
sess.close()
print(a)

