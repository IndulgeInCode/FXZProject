# -*- coding: utf-8 -*-

import os
import dbConnect
import re

r_imdb = '\,|\.|\!|\(|\)|\"|<br|/>'
def getLength(data):
    line = re.sub(r_imdb,'', data)
    result = line.split(' ')
    # allSentences.extend(result)
    return line, len(result)

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
        print(len(files))



def listdir(path):  # 传入存储的list
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            continue
        else:
            list_name.append(file_path)

    return list_name

def getDataTrain():
    train_pos = listdir('/Users/fengxuanzhen/pycharm-workspace/practiceLSTM/aclImdb/train/pos')
    train_neg = listdir('/Users/fengxuanzhen/pycharm-workspace/practiceLSTM/aclImdb/train/neg')
    results = []
    for file_dir in train_pos:
        f = open(file_dir)
        txt = f.read()
        txt,len = getLength(txt)
        record = [txt, 3, 3, len]
        f.close()
        results.append(record)
    for file_dir in train_neg:
        f = open(file_dir)
        txt = f.read()
        txt, len = getLength(txt)
        record = [txt, 0, 0, len]
        f.close()
        results.append(record)
    dbConnect.insertDataInList(results, 2)

def getDataTest():
    train_pos = listdir('/Users/fengxuanzhen/pycharm-workspace/practiceLSTM/aclImdb/test/pos')
    train_neg = listdir('/Users/fengxuanzhen/pycharm-workspace/practiceLSTM/aclImdb/test/neg')
    results = []
    for file_dir in train_pos:
        f = open(file_dir)
        txt = f.read()
        txt, len = getLength(txt)
        record = [txt, 3, 3, len]
        f.close()
        results.append(record)
    for file_dir in train_neg:
        f = open(file_dir)
        txt = f.read()
        txt, len = getLength(txt)
        record = [txt, 0, 0, len]
        f.close()
        results.append(record)
    dbConnect.insertDataInList(results, 2)



if __name__ == "__main__":
    # Test batch generator
    # getDataTrain()
    getDataTest()
