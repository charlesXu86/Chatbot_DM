# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dqn_dataprocess.py
 
@Time    :   2019-09-25 17:54
 
@Desc    :   将对话数据处理为适合dqn训练的数据
 
'''

import numpy as np
import pprint

def read_data(file):
    '''
    预览数据
    :param file:
    :return:
    '''
    data = np.load(file, encoding='latin1')
    pprint.pprint(data)

file1 = 'training_data_encode.npy'
file2 = 'mapping.p'
read_data(file2)
