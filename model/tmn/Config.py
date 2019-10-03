# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   Config.py
 
@Time    :   2019-08-30 09:48
 
@Desc    :
 
'''

class parameters(object):

    def __init__(self):
        super(parameters, self).__init__()
        self.train_data = '/home/xsq/nlp_code/Dialogue_management/data/tmn_data.txt'
        self.test_data = "./data/test.ner.pl"
        self.cla_train_data = "./data/train.cla.pl"
        self.cla_test_data = "./data/test.cla.pl"
        self.dict_dir = "./saved_models"
        self.saved_models_dir = "./saved_models"
        self.pretrain_bert_model_dir = "/Data/public/Bert/chinese_L-12_H-768_A-12"
        self.pretrain_xlnet_model_dir = "/Data/public/chinese_xlnet_mid_L-24_H-768_A-12"
        self.embedding_file = '/Data/xiaobensuan/TX50W.vec'
        self.topic_num = 5                  # 这个参数怎么定最好
        self.hidden_num = [500, 500]        # hidden layer size
        self.topic_emb_dim = 150            # topic memory size
        self.max_seq_len = 100
        self.embeds_dim = 64
        self.max_epoch = 800
        self.min_epoch = 50
        self.patient = 10
        self.patient_global = 60
        self.pre_train_epochs = 50
        self.alter_train_epochs = 50
        self.kl_growing_epoch = 0
        self.target_sparsity = 0.75
        self.batch_size = 32
        self.epochs = 50
        self.max_seq_len = 100
        self.shortcut = True
        self.transform = None


if __name__ == "__main__":
    parameters()
