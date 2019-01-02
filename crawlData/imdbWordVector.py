# -*- coding: UTF-8 -*-

from gensim.models import Word2Vec
import dbConnect
import sys  # 提供了许多函数和变量来处理 Python 运行时环境的不同部分.
import jieba
import numpy as np
import re
import matplotlib.pyplot as plt
import hierarchicalAttention as hAT
from tensorflow.contrib.keras import preprocessing

reload(sys)
sys.setdefaultencoding('utf-8')

# 正则过滤表达式
r = "（|）|；|、|！|，|。|\*|？|~|\<|\>|\s+"
r_imdb = ",|.|!|(|)"
maxSeqLength = 250

EMBEDDING_DIM = 10
TRAINTYPE = 1
TESTTYPE = 0

MAXREVLEN = 6
SENTLENGTH = 20

def buildModel():
    # 所有词集合，包括重复词
    # allSentences = []

    data = dbConnect.getData(begin=0, end=22000)

    sentences = []
    for x in data:
        line = re.sub(r_imdb, '', str(x[1]))
        result = line.split(' ')
        # allSentences.extend(result)
        sentences.append(result)

    # min_count指定了需要训练词语的最小出现次数，默认为5
    # size指定了训练时词向量维度，默认为100
    # worker指定了完成训练过程的线程数，默认为1不使用多线程。只有注意安装Cython的前提下该参数设置才有意义
    model = Word2Vec(sentences, min_count = 1, size = EMBEDDING_DIM)

    # 保存模型
    model.save("word2vecModel/word2vecModel")


# 将句子变成向量形式
def getTrainSenteceVec(type):
    if type == TRAINTYPE:
        data = dbConnect.getData(begin = 0, end = 8000, type = 2)
    elif type ==  TESTTYPE:
        data = dbConnect.getData(begin=8000, end=12000, type = 2)

    return getVec(data)

#将获取长数据
def getLongRecord(type):
    if type == TRAINTYPE:
        longData = dbConnect.getData(begin=0, end=9000, type = 2)
    elif type == TESTTYPE:
        longData = dbConnect.getData(begin=9000, end=6000, type = 2)

    X_train, y_train, seq_len_train = getSplitVec(longData)
    X_train_noh, y_train_noh, seq_len_train_noh = getVec(longData)
    return X_train, y_train, seq_len_train, X_train_noh, y_train_noh, seq_len_train_noh



def getVec(data):
    model = Word2Vec.load('word2vecModel/word2vecModel')
    # print (model[u'罗技'])
    x = []
    y = []
    seq_length = []

    # 遍历每条数据，并转换为向量形式
    for row in data:
        sentenVec = np.zeros([maxSeqLength, EMBEDDING_DIM], dtype='float32')
        sentence = row[1]
        se_list = sentence.split(' ')

        count = 0
        for out in se_list:
            if (count < maxSeqLength and out in model):
                sentenVec[count] = model[out]
                count += 1
        seq_length.append(count)
        x.append(sentenVec)

        if(row[3] > 2):
            y.append(1)
        else :
            y.append(0)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(seq_length, dtype=np.int32)

def getSplitVec(data):
    model = Word2Vec.load('word2vecModel/word2vecModel')
    # print (model[u'罗技'])
    x = []
    y = []
    review_len = []

    # 遍历每条数据，并转换为向量形式
    for row in data:
        sentence = hAT.splitSentence(row[1])
        rowVec = []
        count = 0
        for sent in sentence:
            count += 1
            se_list = jieba.cut(sent)
            sentenVec = []
            for out in se_list:
                if (out in model):
                    sentenVec.append(model[out])
            rowVec.append(sentenVec)
        x.append(rowVec)
        review_len.append(min(count, MAXREVLEN))
        if(row[3] > 2):
            y.append(1)
        else :
            y.append(0)
    #对段落化之后的数据进行整理
    x_formate = preprocess_review(x, MAXREVLEN, SENTLENGTH)
    return np.array(x_formate, dtype=np.float32), np.array(y, dtype=np.int32), np.array(review_len, dtype=np.int32)


def preprocess_review(data, max_rev_len, sent_length,  keep_in_dict=10000):
    length = len(data)
    data_formatted = np.zeros([length, max_rev_len, sent_length, EMBEDDING_DIM], dtype='float32')

    for i in range(length):
        for j in range(min(len(data[i]), max_rev_len)):
            for k in range(min(len(data[i][j]), sent_length)):
                data_formatted[i][j][k] = data[i][j][k]
    return data_formatted


def getReviewStatistic():

    x = range(25, 2500, 100)
    y = dbConnect.getLengthStatistic(type = 2)

    y_formate1 = []
    y_formate2 = []
    batch_size= len(y)/100
    for i in range(len(x)):
        y_formate1.append(y[i *batch_size: (i+1)* batch_size])
    for k in range(len(y_formate1)):
        temp = 0;
        for j in range(len(y_formate1[k])):
            temp += y_formate1[k][j]
        y_formate2.append(temp)

    plt.title('IMDB Content length')

    plt.bar(x, y_formate2, width=20, label=u'length', color='#666666')
    plt.show()


if __name__ == '__main__':
    # getContentStatistic()
    # buildModel()
    getReviewStatistic()
