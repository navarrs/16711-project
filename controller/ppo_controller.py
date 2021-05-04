# ------------------------------------------------------------------------------
# @brief    wrapper for a PPO-based controller
# ------------------------------------------------------------------------------
import numpy as np
import torch
import os

from controller.models.policies.seq2seq_policy import Seq2SeqPolicy
from gym import Space
from habitat import logger
from habitat_baselines.utils.common import batch_obs
from habitat.config import Config
from habitat_baselines.rl.ppo import PPO
from typing import Optional, Union


class PPOController():
    def __init__(
        self, config: Config,
        obs_space: Space,
        act_space: Space,
        stop_on_error: bool = True,
    ) -> None:
        """
        RL-based controller
        Args
        ----
            config: yaml file with config params
            obs_space: observation space. currently uses: DEPTH, RGB, POINTGOALS     
            act_space: action space. currently uses MOVE_FORWARD, TURN_LEFT,
                TURN_RIGHT, STOP
            stop_on_error: unused
        """
        self._config = config
        self._obs_space = obs_space
        self._act_space = act_space

        self._device = (
            torch.device("cuda", self._config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not self._config.USE_CPU
            else torch.device("cpu")
        )

        # @TODO: add stop on error ?
        self._stop_on_error = stop_on_error

        self.build_controller()

    def build_controller(self) -> None:
        """
        Sets the controller up
        """
        model_cfg = self._config.MODEL
        model_cfg.defrost()
        model_cfg.TORCH_GPU_ID = self._config.TORCH_GPU_ID
        model_cfg.freeze()

        if model_cfg.POLICY == "seq2seq":
            self._actor_critic = Seq2SeqPolicy(
                observation_space=self._obs_space,
                action_space=self._act_space,
                model_config=model_cfg,
            )
        else:
            logger.error(f"invalid policy {model_cfg.POLICY}")
            raise ValueError

        self._actor_critic.to(self._device)

        ppo = self._config.RL.PPO
        self._agent = PPO(
            actor_critic=self._actor_critic,
            clip_param=ppo.clip_param,
            ppo_epoch=ppo.ppo_epoch,
            num_mini_batch=ppo.num_mini_batch,
            value_loss_coef=ppo.value_loss_coef,
            entropy_coef=ppo.entropy_coef,
            lr=ppo.lr,
            eps=ppo.eps,
            max_grad_norm=ppo.max_grad_norm,
            use_normalized_advantage=ppo.use_normalized_advantage,
        )

        ckpt_dict = torch.load(
            self._config.RL.ppo_checkpoint, map_location="cpu")
        self._agent.load_state_dict(ckpt_dict["state_dict_agent"])
        self._actor_critic = self._agent.actor_critic
        self.reset()

    def reset(self) -> None:
        self._recurrent_hidden_states = torch.zeros(
            self._actor_critic.net.num_recurrent_layers,
            self._config.NUM_PROCESSES,
            self._config.RL.PPO.hidden_size,
            device=self._device
        )

        self._prev_action = torch.zeros(
            self._config.NUM_PROCESSES, 1, device=self._device, dtype=torch.long)

        self._not_done_masks = torch.zeros(
            self._config.NUM_PROCESSES, 1, device=self._device)

    def get_next_action(
        self, observations,
        deterministic: Optional[bool] = False, **kwargs
    ) -> int:
        """
        Computes controller's next action
        Args
        ----
            observations: environment observations. need to match the observation 
                space defined during initialization. 
            deterministic: if True, samples actions sometimes. 
            dones: done episodes
        Return
        ------
            action
        """
        dones = kwargs.get('dones', None)
        if not dones is None:
            self.update_masks(dones)

        batch = batch_obs(observations, device=self._device)
        self._actor_critic.eval()
        with torch.no_grad():
            (
                _,
                action,
                _,
                self._recurrent_hidden_states,
            ) = self._actor_critic.act(
                batch,
                self._recurrent_hidden_states,
                self._prev_action,
                self._not_done_masks,
                deterministic=deterministic,
            )
            self._prev_action = action
        return action.item()

    def update_masks(self, dones) -> None:
        self._not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=self._device,
        )
