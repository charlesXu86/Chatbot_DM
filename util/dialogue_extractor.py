# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dialogue_extractor.py
 
@Time    :   2019-09-18 11:13
 
@Desc    :   从文本数据中提取对话
 
'''

import re
import json
import requests

ai_cus_reg = re.compile('AI:\[(.+?)\]用户:\[(.+?)\]')

spilt_list = ['开场白：\\n', '逾期提醒：\\n', '哪里的：\\n', '']

def get_cuishou(file, url, file2):
    '''
    step1:提取原始数据
    :param file:
    :param url:
    :param file2:
    :return:
    '''
    file2 = open(file2, "w", encoding='utf-8')

    with open(file, encoding='utf-8') as f:
        row_data = f.readlines()
        for row in row_data:
            id, orderId, text = row.strip().split('\t')
            if text is not None:

                res = requests.post(url, json={"msg": text})
                print(res)

                prob2 = json.loads(res.content).get('problem', '-1')
                file2.write(orderId + "\t" + prob2 + "\t" + text + "\n")


def extract_data(file1, file2):
    '''
    step2: 转换成 Q：
                 A：   格式
    :param file1:
    :param flie2:
    :return:
    '''
    f2 = open(file2, 'w', encoding='utf-8')
    with open(file1, 'r', encoding='utf-8') as f1:
        row_data = f1.readlines()
        for row in row_data:
            tel, prob, texts = row.split('\t')
            alldia = ai_cus_reg.findall(texts)

            for i in range(len(alldia)):
                q = 'Q:'
                a = 'A:'
                qs_be = alldia[i][0]
                qs = qs_be.strip().split(']AI:')[0]      # Q
                # if '开场白：\\n' in qs:
                #     qs = qs.split('开场白：\\n')[1]


                # qs = qs_be.strip().split('\\n')[1]
                aa = alldia[i][1].strip()       # A
                f2.write(q + '\t' + qs + '\n' + a + '\t' + aa + '\n')
            f2.write('\n')
            print(tel)

def clean_QA_data(msg):
    '''
    step3:清洗QA数据
    :param file1:
    :param file2:
    :return:
    '''
    qs = 1


# step1
# file = '/Users/charlesxu/PycharmProjects/practice/data/0905.csv'
# url = 'http://172.18.103.43:8000/api/payback_class'
# file2 = 'result3.txt'
# get_cuishou(file, url, file2)


# step2
file = '/home/xsq/nlp_code/Dialogue_management/data/result3.txt'
file2 = '/home/xsq/nlp_code/Dialogue_management/data/dialogue0925.txt'
extract_data(file, file2)


# step3


