# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   __init__.py.py
 
@Time    :   2019-09-27 11:40
 
@Desc    :
 
'''

# we need to import the policy first
from chatbot_dm.policies.policy import Policy

pass
# and after that any implementation
from chatbot_dm.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from chatbot_dm.policies.embedding_policy import EmbeddingPolicy
from chatbot_dm.policies.fallback import FallbackPolicy
from chatbot_dm.policies.keras_policy import KerasPolicy
from chatbot_dm.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from chatbot_dm.policies.sklearn_policy import SklearnPolicy
from chatbot_dm.policies.form_policy import FormPolicy
from chatbot_dm.policies.two_stage_fallback import TwoStageFallbackPolicy
from chatbot_dm.policies.mapping_policy import MappingPolicy
