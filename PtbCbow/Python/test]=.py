#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:31:56 2016

@author: robbe
"""

import tensorflow as tf
import reader

def length_of_seq(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used,reduction_indices=1)
    length = tf.reduce_mean(length)
    length = tf.cast(length, tf.int32)
    return length

num_step = 8
batch_size = 6
num_history = 5 
raw_data = '/home/robbe/SpeechHub/PtbCbow/Input'
a = reader.ptb_raw_data(raw_data)
x,y,z = reader.ptb_producer(a[1],batch_size,num_step,num_history)



a = tf.constant([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]);
b = tf.reduce_sum(a,0,keep_dims=True)

sequence = length_of_seq(z)

session = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(session, coord=coord)

out1,out2,out3,c,d = session.run([x,y,z,b,sequence])
print('round1')
print(out1)
print(out2)
print(out3)
print(c)
print(d)
print('round2' )
out1,out2,out3,c,d = session.run([x,y,z,b,sequence])
print(out1)
print(out2)
print(out3)
print(c)
print(d)
out1,out2,out3,c,d = session.run([x,y,z,b,sequence])
print(out1)
print(out2)
print(out3)
print(c)
print(d)

