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
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();

    for value in data:
        # id = generate_gid()
        cursor.execute('insert into review (content, score, sentiment) values (%s, %s, %s)',[value["content"], value["score"], value["sentiment"]])
        print cursor.rowcount
        conn.commit()
    cursor.close()

def getData(begin, end):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    sqlsearch = "select * from review limit "+ str(begin) +", "+ str(end)
    cursor.execute(sqlsearch)
    values = cursor.fetchall()
    cursor.close()
    return values

def getLongData(begin, end):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    sqlsearch = "select * from review where char_length(content) > 70 limit "+ str(begin) +", "+ str(end)
    cursor.execute(sqlsearch)
    values = cursor.fetchall()
    cursor.close()
    return values

def getLengthStatistic():
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    number = []
    for i in range(500):
        sqlsearch = "select count(*) from review where char_length(content)="+str(i)
        cursor.execute(sqlsearch)
        value = cursor.fetchall()
        number.extend(value[0])
    cursor.close()
    return number

#
# create table review (
#     id bigint(16) AUTO_INCREMENT ,
#     content text(300),
#     score int(3),
#     sentiment int(3),
#     PRIMARY KEY (id)
# );