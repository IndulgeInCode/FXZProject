# -*- coding: UTF-8 -*-

import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):

    #判断结果是不是双向网络的结果，如果是合并
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        #在第三维上进行合并
        inputs = tf.concat(inputs, 2)

    #和bidirectional_dynamic_rnn的time_major设置有关系
    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # tf.tensordot当axes=1时为矩阵乘法；
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape（）

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #（1 * 250 * 300）
    a = inputs * tf.expand_dims(alphas, -1)
    # 按行求和（1 * 300）
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
