# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   model_data.py
 
@Time    :   2020/6/26 9:36 下午
 
@Desc    :   数据特征预处理
 
"""

import logging

import numpy as np
import scipy.sparse
import tensorflow as tf

from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Text, List, Tuple, Any, Union, Generator, NamedTuple
from collections import defaultdict
from chatbot_dm.utils.tf.constants import BALANCED, SEQUENCE

logger = logging.getLogger(__name__)

# Mapping of feature name to a list of numpy arrays representing the actual features
# For example:
# "text_features" -> [
#   "numpy array containing dense features for every training example",
#   "numpy array containing sparse features for every training example"
# ]
Data = Dict[Text, List[np.ndarray]]


class FeatureSignature(NamedTuple):
    """Stores the shape and the type (sparse vs dense) of features."""

    is_sparse: bool
    shape: List[int]


class RasaModelData:
    """

    """