# -*- coding: UTF-8 -*-

import tensorflow as tf
import re
import dbConnect
import numpy as np

MAXLENGTH = 40
def splitSentence(sentence):
    r = re.compile(u"\n|。|\s")

    sentence = re.split(r, sentence)

    sentence_len = len(sentence)
    result = []
    flag = True
    str = ''
    for i in range(sentence_len):
        flag = True
        if(len(str) < MAXLENGTH):
            str += sentence[i]+ " "
        else:
            result.append(str)
            str = sentence[i]
            flag = False
    if flag :
        result.append(str)

    if len(result) == 1 and len(result[0]) >= 100:
        for i in range(len(result[0]) / MAXLENGTH):
            result.append(result[0][i * MAXLENGTH : (i+1) * MAXLENGTH])
        result.pop(0)
    return result

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
    #（256 * 250 * 1）
    a = inputs * tf.expand_dims(alphas, -1)
    # 按行求和（1 * 300）
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def seq_attention(inputs, attention_size, sequence_size=None, time_major=False, return_alphas=False):
    # 判断结果是不是双向网络的结果，如果是合并
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        # 在第三维上进行合并
        inputs = tf.concat(inputs, 2)

    # 和bidirectional_dynamic_rnn的time_major设置有关系
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
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape（）

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # （256 * 250 * 1）
    a = inputs * tf.expand_dims(alphas, -1)
    # 按行求和（1 * 300）
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


if __name__ == '__main__':
    value = dbConnect.getRecordById(975)
    result = splitSentence(value[0][1])

    print result


