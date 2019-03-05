#!/usr/bin/python
# -*- coding: UTF-8 -*-

import mysql.connector
import time, random
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

host = "10.103.240.156"
# host = "localhost"
user = "root"
password = "fengxuanzhen"
database = "sentimentJD_review"
table = "review"
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


def insertDataInList(data, type = 0):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    if type == 0:
        for value in data:
            # id = generate_gid()
            cursor.execute('insert into review (content, score, sentiment) values (%s, %s, %s)',[value["content"], value["score"], value["sentiment"]])
            print cursor.rowcount
            conn.commit()
    else :
        for value in data:
            # id = generate_gid()
            cursor.execute('insert into imdb_review (content, score, sentiment, length) values (%s, %s, %s, %s)',[value[0], value[1], value[2], value[3]])
            print cursor.rowcount
            conn.commit()
    cursor.close()

def getData(begin, end, type = 0):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    if type == 0 :
        sqlsearch = "select * from review limit "+ str(begin) +", "+ str(end)
        cursor.execute(sqlsearch)
        values = cursor.fetchall()
    else:
        sqlsearch = "select * from imdb_review limit " + str(begin) + ", " + str(end)
        cursor.execute(sqlsearch)
        values = cursor.fetchall()
    cursor.close()
    return values

def getLongData(begin, end):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    # sqlsearch = "select * from (select * from review where char_length(content) >= 20 limit "+ str(begin) +", "+ str(end)+" ) da order by char_length(da.content) desc"
    sqlsearch = "select * from review where char_length(content) > 20 limit "+ str(begin) +", "+ str(end)
    cursor.execute(sqlsearch)
    values = cursor.fetchall()
    cursor.close()
    return values


def getLengthStatistic(type = 0):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();
    number = []
    if type == 0:
        for i in range(500):
            sqlsearch = "select count(*) from review where char_length(content)="+str(i)
            cursor.execute(sqlsearch)
            value = cursor.fetchall()
            number.extend(value[0])
    else:
        for i in range(2500):
            sqlsearch = "select count(*) from imdb_review where length="+str(i)
            cursor.execute(sqlsearch)
            value = cursor.fetchall()
            number.extend(value[0])
    cursor.close()
    return number

def getRecordById(id):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();

    sqlsearch = "select * from review where id ="+str(id)
    cursor.execute(sqlsearch)
    value = cursor.fetchall()
    cursor.close()
    return value

def getRecordByContent(content):
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database, auth_plugin=auth_plugin)
    cursor = conn.cursor();

    sqlsearch = "select * from review where content ="+content
    cursor.execute(sqlsearch)
    value = cursor.fetchall()
    cursor.close()
    return value




# create review (
#     id bigint(16),
#     creat_time data,
#     product_name varchar(100),
#     content text(300),
#     score int(3),
#     sentiment int(3),
#     length int(3),
#     token varchar(15),
#     PRIMARY KEY (id)
# );