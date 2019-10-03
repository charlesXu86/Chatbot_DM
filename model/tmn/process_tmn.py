'''
@Author  :   Xu

@Software:   PyCharm

@File    :   tmn_model.py

@Time    :   2019-08-30 10:11

@Desc    :   训练数据预处理

'''

import numpy as np
import gensim
import os
import sys
from scipy import sparse
import pickle
import json
from gensim.parsing.preprocessing import STOPWORDS
import logging
import pathlib
import jieba
import copy

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)

data_file = basedir + '/data/tmn_data.txt'
data_dir = os.path.dirname(data_file)

# 编码格式处理
with open(os.path.join(data_file), 'U') as fin:
    text = gensim.utils.to_unicode(fin.read(), 'latin1').strip()

news_lst = text.split("\n")
print(news_lst[:5])


msgs = []     # 保存所有的query
labels = []   # 标签
label_dict = {}

for n_i, line in enumerate(news_lst):
    label, msg = line.strip().split('\t')
    msg = list(jieba.cut(msg, cut_all = False))   # 应该不分词
    msgs.append(msg)
    if label not in label_dict:
        label_dict[label] = len(label_dict)
    labels.append(label_dict[label])

# 建立语料特征
dictionary = gensim.corpora.Dictionary(msgs)

bow_dictionary = copy.deepcopy(dictionary)
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
bow_dictionary.filter_extremes(no_below=3, keep_n=None)
bow_dictionary.compactify()

def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []

    # build bow
    row = []
    col = []
    value = []
    row_id = 0
    m_labels = []

    # 迭代所有的query
    for d_i, doc in enumerate(text_doc):
        if len(bow_dictionary.doc2bow(doc)) < 3:    # filter too short
            continue
        for i, j in bow_dictionary.doc2bow(doc):
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1

        wids = list(map(seq_dictionary.token2id.get, doc))
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids)
    lens = list(map(len, seq_doc))
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))
    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels

seq_title, bow_title, label_title = get_wids(msgs, dictionary, bow_dictionary, labels)

# shuffle data
indices = np.arange(len(seq_title))
np.random.shuffle(indices)
seq_title = np.array(seq_title)[indices]

# split data
nb_test_samples = int(0.2 * len(seq_title))
seq_title_train = seq_title[:-nb_test_samples]
seq_title_test = seq_title[-nb_test_samples:]


bow_title = bow_title.tocsr()
bow_title = bow_title[indices]
bow_title_train = bow_title[:-nb_test_samples]
bow_title_test = bow_title[-nb_test_samples:]

label_title = np.array(label_title)[indices]
label_title_train = label_title[:-nb_test_samples]
label_title_test = label_title[-nb_test_samples:]

# save
logging.info("save data...")
pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))
pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))
pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))
dictionary.save(os.path.join(data_dir, "dataDictSeq"))
bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("done!")
