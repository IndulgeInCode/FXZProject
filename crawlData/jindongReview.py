#--*--coding:utf-8--*--
import urllib2
import json
import sys
from bs4 import BeautifulSoup
import dbConnect
import time

reload(sys)
sys.setdefaultencoding('utf8')


def getReview(productId = 0):
    # 因为可能会获取 utf-bow 编码的response，所以不能解析为json。若错误，获取错误并跳出
    for i in range(0,10):
        for repetTime in range(5):
            try:
                type = 1 if i%2 == 0 else 3
                url='https://club.jd.com/comment/productPageComments.action?productId='+ productId +'&score='+ str(type) +'&sortType=5&page='+str(i)+'&pageSize=10&isShadowSku=0&fold=1'

                #实现爬多页
                print url
                request=urllib2.Request(url)
                response=urllib2.urlopen(request)
                html =response.read().decode('GBK')

                b = json.loads(html)

                list = []
                for k in b['comments']:
                    dict = {}
                    dict["id"] = k["id"]
                    # 评论内容
                    content = k["content"].encode('utf-8')
                    dict["content"] = content
                    # 创建时间
                    dict["creatTime"] = k["creationTime"].encode('utf-8')
                    dict["isMobile"] = k["isMobile"].encode('utf-8')
                    # 客户端来源
                    dict["userClientShow"] = k["userClientShow"].encode('utf-8')
                    # 购买商品名称
                    dict["referenceName"] = k["referenceName"].encode('utf-8')
                    #购买商品时间
                    referenceTime=k["referenceTime"].encode('utf-8')
                    dict["referenceTime"] = referenceTime
                    score = k["score"]
                    dict["score"] = score

                    if(score <= 2):
                        dict["sentiment"] = 0.0
                    else:
                        dict["sentiment"] = 4.0
                    list.append(dict)
                dbConnect.insertDataInList(list)

                break
            except:
                print "获取数据错误,重新获取"



def getProductIdByCategre(cat):
    ids = []
    url = 'https://list.jd.com/list.html?cat='+ str(cat)
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    allList = soup.find_all('div', attrs={'class': 'gl-i-wrap j-sku-item'})
    for a in allList:
        ids.append(a['data-sku'])

    return ids


if __name__ == '__main__':
    # ids = getProductIdByCategre('9987,653,655&page=1')
    # ids.extend(getProductIdByCategre('670,671,672&page=1'))
    ids = getProductIdByCategre('670,686,690&page=1')
    ids.extend(getProductIdByCategre('737,794,798&page=1'))
    for id in ids:
        getReview(id)


