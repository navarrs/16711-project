# ------------------------------------------------------------------------------
# @brief    wrapper for a PPO-based controller
# ------------------------------------------------------------------------------
import numpy as np
import torch
import os

from controller.models.policies.seq2seq_policy import Seq2SeqPolicy
from gym import Space
from habitat import logger
from habitat.config import Config
from habitat_baselines.rl.ppo import PPO
from typing import Optional, Union


class PPOController():
    def __init__(
        self,
        config: Config,
        obs_space: Space,
        act_space: Space,
        stop_on_error: bool = True,
    ) -> None:
        self._config = config
        self._obs_space = obs_space
        self._act_space = act_space

        # @TODO: add stop on error
        self._device = (
            torch.device("cuda", self._config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self._stop_on_error = stop_on_error

        # logger.add_filehandler(self._config.LOG_FILE)
        # logger.info(f"Config:\n{self._config}")
        self._build_follower()

    def _build_follower(self) -> None:
        rl_config = self._config.RL.PPO
        self.setup_actor_critic_agent(self._config.RL)

    def setup_actor_critic_agent(self, rl_config) -> None:
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

        ppo = rl_config.PPO
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
        
        ckpt_dict = torch.load(rl_config.ppo_checkpoint, map_location="cpu")
        # self._actor_critic.load_state_dict(ckpt_dict["state_dict_ac"])
        self._agent.load_state_dict(ckpt_dict["state_dict_agent"])
        self._actor_critic = self._agent.actor_critic

        # logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
        # logger.info("Finished setting up actor critic model.")
        
    def get_device(self) -> torch.device:
        return self._device

    def get_actor_critic(self):
        return self._actor_critic

    def get_agent(self):
        return self._agent

    def get_next_action(
        self, batch, recurrent_hidden_states,
        prev_action, not_done_masks, deterministic=False
    ) -> Optional[Union[int, np.array]]:
        self._actor_critic.eval()
        with torch.no_grad():
            (
                _,
                action,
                _,
                recurrent_hidden_states,
            ) = self._actor_critic.act(
                batch,
                recurrent_hidden_states,
                prev_action,
                not_done_masks,
                deterministic=deterministic,
            )
        return action, recurrent_hidden_states
