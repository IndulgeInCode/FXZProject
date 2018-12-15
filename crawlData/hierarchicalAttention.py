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


def attention(atten_inputs, atten_size):
    ## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
    print('attention inputs: ' + str(atten_inputs))
    max_time = int(atten_inputs.shape[1])
    print("max time length: " + str(max_time))
    combined_hidden_size = int(atten_inputs.shape[2])
    print("combined hidden size: " + str(combined_hidden_size))
    W_omega = tf.Variable(tf.random_normal([combined_hidden_size, atten_size], stddev=0.1, dtype=tf.float32))
    b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))
    u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32))

    v = tf.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, combined_hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    print("v: " + str(v))
    # u_omega is the summarizing question vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    print("vu: " + str(vu))
    exps = tf.reshape(tf.exp(vu), [-1, max_time])
    print("exps: " + str(exps))
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    print("alphas: " + str(alphas))
    atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
    print("atten outs: " + str(atten_outs))
    return atten_outs, alphas

def showValue(input):
    print("demention is ", input.shape)


if __name__ == '__main__':
    value = dbConnect.getRecordById(975)
    result = splitSentence(value[0][1])

    print result


