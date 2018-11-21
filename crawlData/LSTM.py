# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import wordVector
# import matplotlib.pyplot as plt


TIME_STEPS=10
BATCH_SIZE=100
HIDDEN_UNITS1=10
HIDDEN_UNITS2=10
LEARNING_RATE=0.0001
EPOCH=50
# 最后输出分类类别数量
class_num = 1
layer_number = 8

TRAIN_EXAMPLES=400
TEST_EXAMPLES=400

#------------------------------------产生数据-----------------------------------------------#

# get values
# one-hot on labels

X_train, y_train= wordVector.getTrainSenteceVec()

X_test = X_train

TRAIN_EXAMPLES = len(X_train)

#trans the shape to (batch,time_steps,input_size)
#
X_train=np.reshape(X_train,newshape=(-1,TIME_STEPS,1))
y_train=np.reshape(y_train,newshape=(-1,1))
X_test=np.reshape(X_test,newshape=(-1,TIME_STEPS,1))





#print(X_train.shape)
#print(y_dummy.shape)
#print(X_test.shape)

#-----------------------------------------------------------------------------------------------------#


#--------------------------------------定义 graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------构建 LSTM------------------------------------------#
    #place hoder
    X_tr=tf.placeholder(dtype=tf.float32,shape=[None,10, 1],name="input_placeholder")

    y_tru=tf.placeholder(dtype=tf.float32 ,shape=[None,class_num],name="pred_placeholder")

    #lstm instance
    lstm_forward=rnn.LSTMCell(num_units=HIDDEN_UNITS1
                              # ,num_proj=1
                              )

    mlstm_cell = rnn.MultiRNNCell([lstm_forward] * layer_number, state_is_tuple=True)

    lstm_backward=rnn.LSTMCell(num_units=HIDDEN_UNITS2
                              # ,num_proj = 1
                               )
    mlstm_cell = rnn.MultiRNNCell([lstm_backward] * layer_number, state_is_tuple=True)

    outputs,states=tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_forward,
        cell_bw=lstm_backward,
        inputs=X_tr,
        dtype=tf.float32
    )

    outputs_fw=outputs[0]
    outputs_bw = outputs[1]
    h=outputs_fw[:,-1,:]+outputs_bw[:,-1,:]
    # print("h.shape is :",h.shape)

#深度LSTM网络
    # lstm_cell1 = rnn.LSTMCell(num_units=HIDDEN_UNITS1)
    # multi_lstm = rnn.MultiRNNCell(cells=l[lstm_cell] * layer_number, state_is_tuple=True)
    # # initialize to zero
    # init_state = multi_lstm.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    # # dynamic rnn
    # outputs, states = tf.nn.dynamic_rnn(cell=multi_lstm, inputs=X_p, initial_state=init_state, dtype=tf.float32)
    # # print(outputs.shape)
    # h = outputs[:, -1, :]
    #---------------------------------------;-----------------------------------------------------#

    #--------------------------------定义损失和优化函数----------------------------------#

    W = tf.Variable(tf.truncated_normal([HIDDEN_UNITS1, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNITS1]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h, W) + bias)

    cross_entropy = -tf.reduce_mean(y_tru * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(y_pre, y_tru)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



    #print(loss.shape)

    # correct_prediction = tf.equal(y_pre, y_tru)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    # cross_loss=tf.losses.softmax_cross_entropy(onehot_labels = y_tru,logits=y_pre)
    # optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss)

    init=tf.global_variables_initializer()




#-------------------------------------------定义 Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)

    print ("X_train.shape :", X_train.shape)
    print ("y_train.shape :", y_train.shape)

    print "循环",TRAIN_EXAMPLES/BATCH_SIZE,"次"

    for epoch in range(1,EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses=[]
        accus=[]
        #test_losses=[]

        print "epoch:",epoch

        #range最大为7？？？？？？
        for j in range(7):
            accu=sess.run(
                    fetches=(accuracy),
                    feed_dict={
                            X_tr:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_tru:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            # train_losses.append(train_loss)
            accus.append(accu)
        # print "average training loss:", sum(train_losses) / len(train_losses)
        print "accuracy:",sum(accus)/len(accus)
        print "\n"

