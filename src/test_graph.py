import tensorflow as tf
import numpy as np

def aaaa(input_):
    with tf.name_scope('B') as scope:
        a = tf.get_variable(name='a', shape=[10,10])
        print(a)
    b = tf.reduce_sum(input_)
    print(b)
    return b

def bbbb(input_, out_):
    with tf.name_scope('C') as scope:
        a = tf.get_variable(name='a', shape=[10,10])
        print(a)
        b = tf.reduce_sum(input_ + out_)
        print(b)
        return b


with tf.variable_scope('A') as scope:
    a = tf.placeholder(dtype=tf.float32, shape=[10], name='a')
    b = tf.placeholder(dtype=tf.float32, shape=[10], name='b')
    print(a)
    print(b)
    aa = aaaa(a)
    tf.get_variable_scope().reuse_variables()
    bb = bbbb(a,b)
    # aaa = aaaa(a)
    # assert a_1 == a_2

# with tf.Session() as sess:
#     feed_dict = {a: np.zeros(shape=[10])}
#     sess.run(aa, feed_dict=feed_dict)
#     sess.run(aaa, feed_dict=feed_dict)
#     feed_dict = {a: np.zeros(shape=[10]),b: np.zeros(shape=[10])}
#     sess.run(bb, feed_dict=feed_dict)
    
