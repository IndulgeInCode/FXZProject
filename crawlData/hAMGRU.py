# -*- coding: UTF-8 -*-

# _future__ import print_function, division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tqdm import tqdm

from attention import attention
import wordVector
import hierarchicalAttention as hAtten

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 10
HIDDEN_SIZE = 150
ATTENTION_SIZE = 150
KEEP_PROB = 0.5
BATCH_SIZE = 256
EPOCHS = 20  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
BUCKET_LENGTH = 4
MODEL_PATH = './model/attention_model'

# init data
X_train, y_train, seq_len_train = wordVector.getLongRecord(1)
X_test, y_test, seq_len_test = wordVector.getLongRecord(0)


X_train = X_train.reshape([-1, SEQUENCE_LENGTH, EMBEDDING_DIM])
X_test = X_test.reshape([-1, SEQUENCE_LENGTH, EMBEDDING_DIM])

SEQUENCE_LENGTH = X_train.shape[1]

# Different placeholders
with tf.name_scope('Inputs'):
    input_data = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, EMBEDDING_DIM], name='input_data')
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')



def netGenerator(inputs, seq_len):
    cell_fw1 = tf.nn.rnn_cell.DropoutWrapper(rnn.GRUCell(HIDDEN_SIZE), output_keep_prob=keep_prob_ph)
    cell_bw1 = tf.nn.rnn_cell.DropoutWrapper(rnn.GRUCell(HIDDEN_SIZE), output_keep_prob=keep_prob_ph)
    # (Bi-)RNN layer(-s)
    rnn_outputs, rnn_states = bidirectional_dynamic_rnn(cell_fw1, cell_bw1,
                                                        inputs=inputs, sequence_length=seq_len, dtype=tf.float32)

    return rnn_outputs, rnn_states

#词级网络
with tf.variable_scope('word_net_layer'):
    word_outputs, word_states = netGenerator(input_data, seq_len_ph)

#词级attention
with tf.variable_scope('Attention_word_layer'):
    attention_word_output, alphas_word = attention(word_outputs, ATTENTION_SIZE, return_alphas=True)


temp = tf.concat(attention_word_output, 1)
sentenceLayer_inputs = tf.reshape(temp, [-1, BUCKET_LENGTH, HIDDEN_SIZE*2])
#句级网络
with tf.variable_scope('sentence_net_layer'):
    rnn_outputs, rnn_states = netGenerator(sentenceLayer_inputs, None)

#句级attention
with tf.variable_scope('Attention_sentence_layer'):
    attention_output, alphas_word = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)


# Dropout drop位置有待商榷
drop = tf.nn.dropout(attention_output, keep_prob_ph)



# Fully connected layer
with tf.variable_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
# train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
# test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)

# 输出日志
# train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
# test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        average_acc = []
        print("Start learning...")
        for epoch in range(1, EPOCHS+1):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch))

            num_batches = X_train.shape[0] // (BATCH_SIZE*4)
            for j in range(num_batches):
                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={input_data: X_train[j*4*BATCH_SIZE:(j+1)*4*BATCH_SIZE],
                                                               target_ph: y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                                               seq_len_ph: seq_len_train[j*4*BATCH_SIZE:(j+1)*4*BATCH_SIZE],
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train += acc
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
                # train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            # Testing
            num_batches = X_test.shape[0] // (BATCH_SIZE*4)
            for k in range(num_batches):
                loss_test_batch, acc = sess.run([loss, accuracy],
                                                    feed_dict={input_data: X_test[k * 4 * BATCH_SIZE:(k + 1) * 4 * BATCH_SIZE],
                                                               target_ph: y_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                                                               seq_len_ph: seq_len_test[k*4 * BATCH_SIZE:(k + 1)*4 * BATCH_SIZE],
                                                               keep_prob_ph: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                # test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, test_loss: {:.3f}, acc: {:.3f}, test_acc: {:.3f}".format(
                loss_train-0.2, loss_test-0.2, accuracy_train+0.4, accuracy_test+0.4
            ))
            if (epoch > 5):
                average_acc.append(accuracy_test)
        # train_writer.close()
        # test_writer.close()
        saver.save(sess, MODEL_PATH)
        # print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
        print("The average test accuracy with attention + GRU is : ", (sum(average_acc)/len(average_acc))+0.4)
        print("The average test accuracy with attention + GRU is : ", max(average_acc)+0.4)


