# ------------------------------------------------------------------------------
# @brief    controller base class
# ------------------------------------------------------------------------------
import numpy as np
import torch
import os

from controller.models.policies.seq2seq_policy import Seq2SeqPolicy
from controller.ppo_controller import PPOController
from controller.simple_controller import SimpleController
from enum import Enum
from gym import Space
from habitat import logger
from habitat.config import Config
from habitat_baselines.rl.ppo import PPO
from typing import Optional, Union

from controller.simple_unit_controller import SimpleUnitController

SUPPORTED_CONTROLLERS = ["simple_controller", "ppo_controller", "simple_unit_controller"]

class ControllerType(Enum):
    BLACKBOX = 0
    FALLBACK = 1

class BaseController():
    def __init__(self, config: Config, *args, **kwargs) -> None:
        """
        Simple base class to create controllers
        Args
        ----
            config: yaml file containing config params to create the controller
            controller_type: currently has to be: [BLACKBOX, FALLBACK]
        """
        self._config = config
        self._obs_space = kwargs.get('obs_space', None)
        self._act_space = kwargs.get('act_space', None)
        self._sim = kwargs.get('sim', None)
        
    def build_controller(self, controller_type: ControllerType) -> None:
        
        if controller_type == ControllerType.BLACKBOX:
            controller_id = self._config.CONTROLLERS.blackbox_id
        else:
            controller_id = self._config.CONTROLLERS.fallback_id
        
        assert controller_id in SUPPORTED_CONTROLLERS, \
            f"controller: {controller_id} not in supported controllers: {SUPPORTED_CONTROLLERS}"
        
        if controller_id == "ppo_controller":
            controller = PPOController(
                self._config, self._obs_space, self._act_space)
        elif controller_id == "simple_controller":
            controller = SimpleController(
                self._config, self._sim)
        elif controller_id == "simple_unit_controller":
            controller = SimpleUnitController(
                self._config, self._sim)
        
        logger.info(
            f"Initialized {controller_type} with id: {controller_id}")
        return controller    
