#!/usr/bin/python
# -*- coding: UTF-8 -*-

import mysql.connector
import time, random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

host = "10.103.247.186"
# host = ""
user = "root"
password = "fengxuanzhen"
database = "sentimentJD_review"
auth_plugin="mysql_native_password"
def generate_gid():
    first = int(time.time())
    second = (int)(random.random()*1000)
    strid = str(first) + str(second)
    return int(strid)

def connectSql(user, password, host):
    try:
        cnsql = mysql.connector.connect(user=user, password=password, host=host, database="sentimentJD_review")

    finally:
        cnsql.close()


def insertDataInList(data):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
    cursor = conn.cursor();

    for value in data:
        # id = generate_gid()
        cursor.execute('insert into review (content, score, sentiment) values (%s, %s, %s)',[value["content"], value["score"], value["sentiment"]])
        print cursor.rowcount
        conn.commit()
    cursor.close()

def getTrainData(begin, end):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    sqlsearch = "select * from review limit "+ str(begin) +", "+ str(end)
    cursor.execute(sqlsearch)
    values = cursor.fetchall()
    cursor.close()
    return values

def getTestData(data):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin, charset="utr8")
    cursor = conn.cursor();
    cursor.execute("select * from review limit 0,100 ")
    values = cursor.fetchall()
    cursor.close()
    return values

#
# create table review (
#     id bigint(16) AUTO_INCREMENT ,
#     content text(300),
#     score int(3),
#     sentiment int(3),
#     PRIMARY KEY (id)
# );