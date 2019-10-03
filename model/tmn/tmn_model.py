# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   tmn_model.py
 
@Time    :   2019-08-30 10:11
 
@Desc    :
 
'''

import tensorflow as tf
import numpy as np

from . Config import parameters

class TMN_Model():

    def __init__(self, parameters):
        self.bert_model = parameters.pretrain_bert_model_dir
        self.data_dir = parameters.train_data
        self.topic_num = parameters.topic_num
        self.hidden_num = parameters.hidden_num
        self.topic_emb_dim = parameters.topic_emb_dim
        self.max_seq_len = parameters.max_seq_len
        self.batch_size = parameters.batch_size
        self.max_epoch = parameters.max_epoch
        self.min_epoch = parameters.min_epoch
        self.patient = parameters.patient
        self.patient_global = parameters.patient_global
        self.pre_train_epochs = parameters.pre_train_epochs
        self.alter_train_epochs = parameters.alter_train_epochs
        self.target_sparsity = parameters.target_sparsity
        self.kl_growing_epoch = parameters.kl_growing_epoch
        self.shortcut = parameters.shortcut
        self.transform = parameters.transform




