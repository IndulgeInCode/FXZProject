# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import wordVector
# import matplotlib.pyplot as plt


TIME_STEPS=10
BATCH_SIZE=100
HIDDEN_UNITS1=150
HIDDEN_UNITS_final=150
LEARNING_RATE=0.3
EPOCH=30
# 最后输出分类类别数量
class_num = 2
layer_number = 3

TRAIN_EXAMPLES=400
TEST_EXAMPLES=400

#------------------------------------产生数据-----------------------------------------------#

# get values
# one-hot on labels

X_train, y_train, senten_len = wordVector.getTrainSenteceVec(1)

X_test = X_train

TRAIN_EXAMPLES = len(X_train)

#trans the shape to (batch,time_steps,input_size)
#
# X_train=np.reshape(X_train,newshape=(-1,TIME_STEPS,1))
# y_train=np.reshape(y_train,newshape=(-1,5))
# X_test=np.reshape(X_test,newshape=(-1,TIME_STEPS,1))





#print(X_train.shape)
#print(y_dummy.shape)
#print(X_test.shape)

#-----------------------------------------------------------------------------------------------------#


#--------------------------------------定义 graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------构建 LSTM------------------------------------------#
    #place hoder
    x_tru=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE, 250, 10],name="input_placeholder")

    y_tru=tf.placeholder(dtype=tf.float32 ,shape=[BATCH_SIZE],name="pred_placeholder")

    #lstm instance
    lstm_forward_1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_forward_2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_forward_3 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_forward_4 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_final)

    lstm_forward = rnn.MultiRNNCell(cells=[lstm_forward_1, lstm_forward_2, lstm_forward_4])

    lstm_backward_1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_backward_2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_backward_3 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_backward_4 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_final)

    lstm_backward = rnn.MultiRNNCell(cells=[lstm_backward_1, lstm_backward_2, lstm_backward_4])


    outputs,states=tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_forward,
        cell_bw=lstm_backward,
        inputs=x_tru,
        dtype=tf.float32
    )
    outputs_fw=outputs[0]
    outputs_bw = outputs[1]
    # h=outputs_fw[:,-1,:]+outputs_bw[:,-1,:]
    h = outputs_fw[:, -1, :]
    print("h.shape is :",h.shape)

    # Fully connected layer
    with tf.name_scope('Fully_connected_layer'):
        W = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS_final, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
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


    # cross_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_tru,logits=y_hat)
    # # print(loss.shape)
    # correct_prediction = tf.equal(y_hat, y_tru)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    # optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss)
    # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).apply_gradients(zip(grads, tvars))  # 将梯度应用于变量



    # cross_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_tru, logits=h)
    #
    # train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_loss)
    #
    # correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_tru, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #print(loss.shape)
    init=tf.global_variables_initializer()


#-------------------------------------------定义 Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)

    print ("X_train.shape :", X_train.shape)
    print ("y_train.shape :", y_train.shape)

    print ("循环",TRAIN_EXAMPLES/BATCH_SIZE,"次")

    for epoch in range(1,EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses=[]
        accus=[]
        #test_losses=[]

        print "epoch:",epoch

        #range最大为7？？？？？？
        for j in range(TRAIN_EXAMPLES/BATCH_SIZE):
            _,hout, accu, train_loss=sess.run(
                    fetches=(optimizer,h, accuracy, loss),
                    feed_dict={
                            x_tru:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_tru:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            # print hout
            train_losses.append(train_loss)
            accus.append(accu)


        print ("average training loss:", (sum(train_losses) / len(train_losses)))
        print ("accuracy:",sum(accus)/len(accus))
        print ("\n")

