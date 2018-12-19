# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import wordVector
from sklearn import metrics
# import matplotlib.pyplot as plt


TIME_STEPS=10
BATCH_SIZE=124
HIDDEN_UNITS=150
LEARNING_RATE=0.3
EPOCH=60
# 最后输出分类类别数量
class_num = 2
layer_number = 3

TRAIN_EXAMPLES=400
TEST_EXAMPLES=400

#------------------------------------产生数据-----------------------------------------------#

# get values
# one-hot on labels

X_train, y_train, senten_len_train = wordVector.getTrainSenteceVec(1)
X_test, y_test, senten_len_test = wordVector.getTrainSenteceVec(0)


TRAIN_EXAMPLES = len(X_train)
TEST_EXAMPLES = len(X_test)


#-----------------------------------------------------------------------------------------------------#


#--------------------------------------定义 graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------构建 LSTM------------------------------------------#
    #place hoder
    x_tru=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE, 250, 10],name="input_placeholder")

    y_tru=tf.placeholder(dtype=tf.float32 ,shape=[BATCH_SIZE],name="pred_placeholder")

    senten_len_batch = tf.placeholder(dtype=tf.int32 ,shape=[BATCH_SIZE],name="senten_len_batch")

    #lstm instance
    lstm_forward_1 = rnn.BasicRNNCell(num_units=HIDDEN_UNITS)
    lstm_forward_2 = rnn.BasicRNNCell(num_units=HIDDEN_UNITS)
    # lstm_forward_3 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)

    lstm_forward = rnn.MultiRNNCell(cells=[lstm_forward_1])

    lstm_backward_1 = rnn.BasicRNNCell(num_units=HIDDEN_UNITS)
    lstm_backward_2 = rnn.BasicRNNCell(num_units=HIDDEN_UNITS)

    lstm_backward = rnn.MultiRNNCell(cells=[lstm_backward_1])


    outputs,states=tf.nn.dynamic_rnn(
        cell=lstm_forward,
        inputs=x_tru,
        sequence_length=senten_len_batch,
        dtype=tf.float32
    )
    #
    h = states[0]

    # Fully connected layer
    with tf.name_scope('Fully_connected_layer'):
        W = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[1]))
        y_hat = tf.nn.xw_plus_b(h, W, b)
        y_hat = tf.squeeze(y_hat)
        tf.summary.histogram('W', W)

#深度LSTM网络
    # lstm_cell1 = rnn.LSTMCell(num_units=HIDDEN_UNITS1)
    # lstm_cell2 = rnn.LSTMCell(num_units=HIDDEN_UNITS_final)
    # multi_lstm = rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2], state_is_tuple=True)
    # # initialize to zero
    # init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    # # dynamic rnn
    # outputs, states = tf.nn.dynamic_rnn(cell=multi_lstm,
    #                                     inputs=x_tru,
    #                                     # initial_state=init_state,
    #                                     dtype=tf.float32)
    # # print(outputs.shape)
    # h = outputs[:, -1, :]
    #---------------------------------------;-----------------------------------------------------#

    #--------------------------------定义损失和优化函数----------------------------------#

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_tru))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), y_tru), tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    y_predict = tf.round(tf.sigmoid(y_hat))


    #print(loss.shape)
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    init=tf.global_variables_initializer()


#-------------------------------------------定义 Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)
    average_acc = []
    for epoch in range(1,EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses=[]
        train_accus=[]
        test_losses=[]
        test_accus=[]
        y_predicts = []

        print "epoch:",epoch

        #range最大为7？？？？？？
        for j in range(TRAIN_EXAMPLES/BATCH_SIZE -1):
            _, accu, train_loss=sess.run(
                    fetches=(optimizer, accuracy, loss),
                    feed_dict={
                            x_tru:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_tru:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            senten_len_batch:senten_len_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]

                        }
            )
            # print states_fw_out
            train_losses.append(train_loss)
            train_accus.append(accu)

        for k in range(TEST_EXAMPLES/BATCH_SIZE-1):
            test_accu, y_predict_outs, test_loss=sess.run(
                    fetches=(accuracy, y_predict, loss),
                    feed_dict={
                            x_tru:X_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE],
                            y_tru:y_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE],
                            senten_len_batch:senten_len_test[k*BATCH_SIZE:(k+1)*BATCH_SIZE]

                        }
            )
            # print states_fw_out
            y_predicts.extend(y_predict_outs)
            test_losses.append(test_loss)
            test_accus.append(test_accu)

        f1_score = metrics.f1_score(y_test[0: (k + 1) * BATCH_SIZE], y_predicts)
        precision = metrics.precision_score(y_test[0: (k + 1) * BATCH_SIZE], y_predicts)
        recall = metrics.recall_score(y_test[0: (k + 1) * BATCH_SIZE], y_predicts)
        print("loss: {:.3f}, test_loss: {:.3f}, acc: {:.3f}, test_acc: {:.3f}, f1_score: {:.3f}, precision: {:.3f}, recall: {:.3f}".format(
            (sum(train_losses) / len(train_losses)), (sum(test_losses) / len(test_losses)),
            (sum(train_accus)/len(train_accus)), (sum(test_accus) / len(test_accus)),
            f1_score, precision, recall
        ))

        if(epoch > 50):
            average_acc.append((sum(test_accus) / len(test_accus)))

    print("The average test accuracy with LSTM is : ", (sum(average_acc) / len(average_acc)))
    print("The average test accuracy with LSTM is : ", max(average_acc))