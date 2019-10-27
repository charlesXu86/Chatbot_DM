# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   __init__.py.py
 
@Time    :   2019-09-27 11:37
 
@Desc    :
 
'''

import logging

import chatbot_dm

from chatbot_dm.train import train
# from chatbot_dm.test import test
# from chatbot_dm.visualize import visualizec

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = chatbot_dm.__version__