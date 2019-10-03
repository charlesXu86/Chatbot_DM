# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   Config.py
 
@Time    :   2019-08-29 14:53
 
@Desc    :
 
'''

class Model_Config(object):
    '''
    docstring for Config
    '''
    def __init__(self):
        super(Model_Config, self).__init__()
        self.ner_train_data = "./data/train.ner.pl"
        self.ner_test_data = "./data/test.ner.pl"
        self.cla_train_data = "./data/train.cla.pl"
        self.cla_test_data = "./data/test.cla.pl"
        self.if_train_ner = True
        self.if_train_cla = False
        self.dict_dir = "./saved_models"
        self.saved_models_dir = "./saved_models"
        self.pretrain_bert_model_dir = "/Data/public/Bert/chinese_L-12_H-768_A-12"
        self.pretrain_xlnet_model_dir = "/Data/public/XLNet/chinese_xlnet_mid_L-24_H-768_A-12"
        self.layerid = 12
        self.ner_layerid = 12
        self.cla_layerid = 12
        self.embeds_dir = None
        self.embeds_dim = 64
        self.ner_embeds_dim = 64
        self.cla_embeds_dim = 128
        self.batch_size = 16
        self.epochs = 50
        self.max_seq_len = 100