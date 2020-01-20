一、Chatbot_DM
==========================

**Chatbot_DM** 是一个基于 `RASA <https://rasa.com/docs/rasa/user-guide/installation/>`_ 的自定义中文语言理解组件，他可以替换掉RASA中本身的nlu模块，可以使nlu的准确率有较大提升，目前
引入了 **bert** ，后续将引入 **xlnet** 。

同时，`RASA <https://rasa.com/docs/rasa/user-guide/installation/>`_ 的自定义组件可以参考 `Custom NLU Components <https://rasa.com/docs/rasa/api/custom-nlu-components/>`_

目前 **Chatbot_DM** 支持的功能有：

    1、bert vector

    2、bert intent

    3、bert slot


二、安装使用
============

1、安装
>>>>>>>>>>>>>>>>>>

.. code:: python

    pip install chatbot_dm

2、使用
>>>>>>>>>>>>>>>>>>>

**Chatbot_NLU** 的使用是在 **config.yml** 文件中修改配置，常见的使用方法请参考：

.. code:: python

    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "chatbot_nlu.extractors.crf_entity_extractor.CRFEntityExtractor"

    # ner
    - name: "chatbot_nlu.extractors.jieba_pseg_extractor.JiebaPsegExtractor"
      part_of_speech: ["nr"]

    - name: "chatbot_nlu.extractors.bilstm_crf_entity_extractor.BilstmCRFEntityExtractor"


    # Word Embedding
    - name: "chatbot_nlu.featurizers.bert_vectors_featurizer.BertVectorsFeaturizer"
      ip: "172.18.103.43"
      port: 5555
      port_out: 5556
      show_server_config: True
      timeout: 10000
      check_version: False





三、Update News
======================

    * 2020.1.7  接入钉钉群，支持主动推送消息、outgoing交互

    * 2020.1.9  接入微信



四、Resources
======================

.. _`Dingtalk_README`: https://github.com/charlesXu86/Chatbot_Help/blob/master/Dingtalk_README.rst
