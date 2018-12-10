# -*- coding: UTF-8 -*-

# _future__ import print_function, division

import numpy as np
import tensorflow as tf
from keras.datasets import imdb
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tqdm import tqdm

from attention import attention
from utils import get_vocabulary_size, fit_in_vocabulary, zero_pad, batch_generator
import wordVector

NUM_WORDS = 10000
INDEX_FROM = 3
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 10
HIDDEN_SIZE = 150
KEEP_PROB = 0.5
BATCH_SIZE = 256
EPOCHS = 100  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
DELTA = 0.5
MODEL_PATH = './model/noattention_model'

# init data
X_train, y_train, seq_len_train = wordVector.getTrainSenteceVec(3)
X_test, y_test, seq_len_test = wordVector.getTrainSenteceVec(2)
TRAIN_EXAMPLES = len(X_train)
TEST_EXAMPLES = len(X_test)



# Different placeholders
with tf.name_scope('Inputs'):
    input_data = tf.placeholder(tf.float32, [None, SEQUENCE_LENGTH, EMBEDDING_DIM], name='input_data')
    target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

cell_fw1 = rnn.GRUCell(HIDDEN_SIZE)
cell_fw2 = rnn.GRUCell(HIDDEN_SIZE)
lstm_forward = rnn.MultiRNNCell(cells=[cell_fw1, cell_fw2])
cell_bw1 = rnn.GRUCell(HIDDEN_SIZE)
cell_bw2 = rnn.GRUCell(HIDDEN_SIZE)
lstm_backward = rnn.MultiRNNCell(cells=[cell_bw1, cell_bw2])

# (Bi-)RNN layer(-s)
rnn_outputs, rnn_states = bidirectional_dynamic_rnn(lstm_forward, lstm_backward,
                        inputs=input_data, sequence_length=seq_len_ph, dtype=tf.float32)


# rnn_states = tf.concat(rnn_states, 2)

states_fw = rnn_states[0]
states_bw = rnn_states[1]

h = tf.concat([states_fw, states_bw], 1)



# Dropout
drop = tf.nn.dropout(h, keep_prob_ph)





# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE*2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    # tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    # tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), target_ph), tf.float32))
    # tf.summary.scalar('accuracy', accuracy)

# merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE)
test_batch_generator = batch_generator(X_test, y_test, BATCH_SIZE)


session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        average_acc = []
        print("Start learning...")
        for epoch in range(1, EPOCHS+1):
            loss_train = []
            loss_test = []
            accuracy_train = []
            accuracy_test = []

            print("epoch: {}\t".format(epoch))

            num_batches = TRAIN_EXAMPLES/BATCH_SIZE
            for j in range(num_batches):
                loss_tr, acc, _ = sess.run([loss, accuracy, optimizer],
                                                    feed_dict={input_data:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                                               target_ph: y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                                               seq_len_ph: seq_len_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                                                               keep_prob_ph: KEEP_PROB})
                accuracy_train.append(acc)
                loss_train.append(loss_tr)
            # accuracy_train /= num_batches

            # Testing
            num_batches = TEST_EXAMPLES/BATCH_SIZE
            for k in range(num_batches):
                loss_te, acc_te = sess.run(fetches=(loss, accuracy),
                                                    feed_dict={input_data: X_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                                                               target_ph: y_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                                                               seq_len_ph: seq_len_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE],
                                                               keep_prob_ph: 1.0})
                accuracy_test.append(acc_te)
                loss_test.append(loss_te)
            # accuracy_test /= num_batches
            # loss_test /= num_batches

            print("loss: {:.3f}, test_loss: {:.3f}, acc: {:.3f}, test_acc: {:.3f}".format(
                (sum(loss_train)/len(loss_train)), (sum(loss_test)/len(loss_test)), (sum(accuracy_train)/len(accuracy_train)), (sum(accuracy_test)/len(accuracy_test))
            ))
            if(epoch > 70):
                average_acc.append((sum(accuracy_test)/len(accuracy_test)))
        saver.save(sess, MODEL_PATH)
        # print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
        print("The average test with GRU accuracy is : ", (sum(average_acc)/len(average_acc)))
        print("The average test with GRU accuracy is : ", max(average_acc))