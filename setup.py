# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   set_up.py

@Time    :   2019-06-06 16:22

@Desc    :   setup

'''
from setuptools import find_packages, setup, convert_path
import pathlib

# Package meta-data.
NAME = 'chatbot-dm'
DESCRIPTION = '基于RASA对话引擎的对话管理自定义模块，自定义对话policy，引入强化学习'
URL = ''
EMAIL = 'charlesxu86@163.com'
AUTHOR = 'xu'
LICENSE = 'MIT'

def _version():
    ns = {}
    with open(convert_path("chatbot_dm/version.py"), "r") as fh:
        exec(fh.read(), ns)
    return ns['__version__']

__version = _version()

with open("README.rst", "r") as fh:
    long_description = fh.read()

required = [
            'keras>=2.2.4',
            ]

setup(name=NAME,
        version=__version,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/x-rst',
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        package_data={'chatbot_dm': ['resource/*.json', '*.rst']},
        install_requires=required,
        license=LICENSE,
        classifiers=['License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Programming Language :: Python :: Implementation :: PyPy'],)
print("Welcome to Chatbot_DM, and Chatbot_DM version is {}".format(__version))
