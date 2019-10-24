# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dqn_policy.py
 
@Time    :   2019-09-30 10:04
 
@Desc    :
 
'''

from  collections import namedtuple
import copy
import json
import logging
import os
import warnings
import numpy as np
import tensorflow as tf
import  typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from chatbot_dm.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer
)
from chatbot_dm.policies.policy import Policy

import pickle

logger = logging.getLogger(__name__)

class DeepQNetPolicy(Policy):

    def __init__(
        self,
        featurizer: Optional[FullDialogueTrackerFeaturizer] = None,
        priority: int = 1,
        encoded_all_actions: Optional[np.ndarray] = None,
        graph: Optional[tf.Graph] = None,
        session: Optional[tf.Session] = None,
        intent_placeholder: Optional[tf.Tensor] = None,
        action_placeholder: Optional[tf.Tensor] = None,
        slots_placeholder: Optional[tf.Tensor] = None,
        prev_act_placeholder: Optional[tf.Tensor] = None,
        dialogue_len: Optional[tf.Tensor] = None,
        x_for_no_intent: Optional[tf.Tensor] = None,
        y_for_no_action: Optional[tf.Tensor] = None,
        y_for_action_listen: Optional[tf.Tensor] = None,
        similarity_op: Optional[tf.Tensor] = None,
        alignment_history: Optional[tf.Tensor] = None,
        user_embed: Optional[tf.Tensor] = None,
        bot_embed: Optional[tf.Tensor] = None,
        slot_embed: Optional[tf.Tensor] = None,
        dial_embed: Optional[tf.Tensor] = None,
        attn_embed: Optional[tf.Tensor] = None,
        copy_attn_debug: Optional[tf.Tensor] = None,
        all_time_masks: Optional[tf.Tensor] = None,
        **kwargs: Any) -> None:
            if featurizer:
                if not isinstance(featurizer, FullDialogueTrackerFeaturizer):
                    raise TypeError(
                        "Passed tracker featurizer of type {}, "
                        "should be FullDialogueTrackerFeaturizer."
                        "".format(type(featurizer).__name__)
                    )
            super(DeepQNetPolicy, self).__init__(featurizer, priority)

            self._load_params(**kwargs)

            self.encoded_all_actions = encoded_all_actions

            self.graph = graph
            self.session = session
            self.a_in = intent_placeholder
            self.b_in = action_placeholder
            self.c_in = slots_placeholder
            self.b_prev_in = prev_act_placeholder
            self._dialogue_len = dialogue_len
            self._x_for_no_intent_in = x_for_no_intent
            self._y_for_no_action_in = y_for_no_action
            self._y_for_action_listen_in = y_for_action_listen
            self.sim_op = similarity_op

            # store attention probability distribution as
            # concatenated tensor of each attention types
            self.alignment_history = alignment_history

            # persisted embeddings
            self.user_embed = user_embed
            self.bot_embed = bot_embed
            self.slot_embed = slot_embed
            self.dial_embed = dial_embed

            self.target_q = tf.placeholder(tf.float32, [None], name="target_q")
            self.reward = tf.placeholder(tf.float32, [None], name="reward")
            self.terminal = tf.placeholder(tf.float32, [None], name="terminal")

            # internal tf instances
            self._train_op = None
            self._is_training = None
            self._loss_scales = None

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._tf_config = self._load_tf_config(config)
        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_attn_params(config)
        self._load_visual_params(config)

    def max_q(self):
        return

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any
    ) -> None:

        logger.debug("Start training DQN policies.")

        np.random.seed(self.random_seed)