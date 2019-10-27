# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   config.py

@Time    :   2019-09-28 15:16

@Desc    :   加载配置文件

'''

import os
import typing
from typing import Optional, Text, List

import chatbot_dm.utils.io

if typing.TYPE_CHECKING:
    from chatbot_dm.policies.policy import Policy


def load(config_file: Optional[Text]) -> List["Policy"]:
    """Load policy data stored in the specified file."""
    from chatbot_dm.policies.ensemble import PolicyEnsemble

    if config_file and os.path.isfile(config_file):
        config_data = chatbot_dm.utils.io.read_config_file(config_file)
    else:
        raise ValueError(
            "You have to provide a valid path to a config file. "
            "The file '{}' could not be found."
            "".format(os.path.abspath(config_file))
        )

    return PolicyEnsemble.from_dict(config_data)
