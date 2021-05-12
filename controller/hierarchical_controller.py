# ------------------------------------------------------------------------------
# @brief    wrapper for a PPO-based controller
# ------------------------------------------------------------------------------
import habitat_sim
import numpy as np
import torch
import os
import magnum as mn 

from controller.models.policies.cont_seq2seq_policy import ContinuosSeq2SeqPolicy
from controller.models.policies.seq2seq_policy import Seq2SeqPolicy
from gym import Space
from habitat import logger
from habitat_baselines.utils.common import batch_obs
from habitat.config import Config
from habitat_baselines.rl.ppo import PPO
from typing import Optional, Union


class HierarchicalController():
    def __init__(
        self, config: Config,
        obs_space: Space,
        act_space: Space,
    ) -> None:
        """
        RL-based + IL-based controller. RL takes high-level discrete actions and
        IL uses those actions as a subgoal. 
        Args
        ----
            config: yaml file with config params
            obs_space: observation space. currently uses: DEPTH, RGB, POINTGOALS     
            act_space: action space. currently uses 
        """
        self._config = config
        self._obs_space = obs_space
        self._act_space = act_space
        self._max_turn_speed = 1.0
        self._device = (
            torch.device("cuda", self._config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not self._config.USE_CPU
            else torch.device("cpu")
        )
        
        self._vel_control = habitat_sim.physics.VelocityControl()
        self._vel_control.controlling_lin_vel = True
        self._vel_control.controlling_ang_vel = True
        self._vel_control.lin_vel_is_local = True
        self._vel_control.ang_vel_is_local = True
        
        self.build_controller()

    def build_controller(self) -> None:
        """
        Sets the controller up. This controller consists of a hierarchical model:
            RL agent: chooses among 4 actions: FORWARD, LEFT, RIGHT, STOP
            IL agent: performs low level actions based on the high level action
        """
        # Set the RL agent
        hi_lvl_cfg = self._config.MODEL_HIGH_LEVEL
        hi_lvl_cfg.defrost()
        hi_lvl_cfg.TORCH_GPU_ID = self._config.TORCH_GPU_ID
        hi_lvl_cfg.freeze()
        if hi_lvl_cfg.POLICY == "seq2seq":
            self._highlevel_controller = Seq2SeqPolicy(
                observation_space=self._obs_space,
                action_space=self._act_space,
                model_config=hi_lvl_cfg,
            )
        else:
            logger.error(f"invalid policy {hi_lvl_cfg.POLICY}")
            raise ValueError

        ppo = self._config.RL.PPO
        self._agent = PPO(
            actor_critic=self._highlevel_controller,
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
        self._highlevel_controller = self._agent.actor_critic
        self._highlevel_controller.to(self._device)
        self._highlevel_controller.eval()
        logger.info(f"High-Level controller ready")
        
        
        lo_lvl_cfg = self._config.MODEL_LOW_LEVEL
        lo_lvl_cfg.defrost()
        lo_lvl_cfg.TORCH_GPU_ID = self._config.TORCH_GPU_ID
        lo_lvl_cfg.freeze()
        if lo_lvl_cfg.POLICY == "cont_seq2seq":
            self._lowlevel_controller = ContinuosSeq2SeqPolicy(
                observation_space=self._obs_space,
                model_config=lo_lvl_cfg,
                num_sub_tasks=self._act_space.n,
                num_actions=2, 
                batch_size=1
            )
        else:
            logger.error(f"invalid policy {hi_lvl_cfg.POLICY}")
            raise ValueError
        
        ckpt_dict = torch.load(
            self._config.IL.checkpoint_path, map_location="cpu")
        self._lowlevel_controller.load_state_dict(
            ckpt_dict["low_level_state_dict"])
        self._lowlevel_controller.to(self._device)
        self._lowlevel_controller.eval()
        logger.info(f"Low-Level controller ready")

        self.reset()

    def reset(self) -> None:
        self._highlevel_recurrent_hidden_states = torch.zeros(
            self._highlevel_controller.net.num_recurrent_layers,
            self._config.NUM_PROCESSES,
            self._config.RL.PPO.hidden_size,
            device=self._device
        )
        
        self._lowlevel_recurrent_hidden_states = torch.zeros(
            self._lowlevel_controller.state_encoder.num_recurrent_layers,
            self._config.NUM_PROCESSES,
            self._config.MODEL_LOW_LEVEL.STATE_ENCODER.hidden_size,
            device=self._device
        )

        self._high_prev_action = torch.zeros(
            self._config.NUM_PROCESSES, 1, device=self._device, dtype=torch.long)
        
        self._low_prev_action = torch.zeros(
            self._config.NUM_PROCESSES, 2, device=self._device, dtype=torch.long)

        self._high_not_done_masks = torch.zeros(
            self._config.NUM_PROCESSES, 1, device=self._device)
        
        self._low_not_done_masks = torch.zeros(
            self._config.NUM_PROCESSES, 2, device=self._device)
    
    def get_next_action(
        self, 
        observations,
        deterministic: Optional[bool] = False, 
        **kwargs
    ):
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
        batch = batch_obs(observations, device=self._device)
        self._highlevel_controller.eval()
        self._lowlevel_controller.eval()
        
        with torch.no_grad():
            # high level controller 
            (
                _,
                action,
                _,
                self._highlevel_recurrent_hidden_states,
            ) = self._highlevel_controller.act(
                batch,
                self._highlevel_recurrent_hidden_states,
                self._high_prev_action,
                self._high_not_done_masks,
                deterministic=deterministic,
            )
            self._high_prev_action = action
            
            # low level controller
            low_batch = (
                batch, 
                self._lowlevel_recurrent_hidden_states, 
                self._low_prev_action, 
                self._low_not_done_masks,
                action
            )
            
            (
                out_vel, 
                out_stop, 
                self._lowlevel_recurrent_hidden_states
            ) = self._lowlevel_controller(low_batch)
            self._low_prev_action = out_vel
        
        
        self._low_not_done_masks = torch.ones(
            self._config.NUM_PROCESSES, 2, device=self._device)
        self._high_not_done_masks = torch.ones(
            self._config.NUM_PROCESSES, 1, device=self._device)
        
        lin_vel = out_vel[:, 0].cpu().numpy()
        ang_vel = -out_vel[:, 1].cpu().numpy()
        ang_vel = np.clip(ang_vel, -self._max_turn_speed, self._max_turn_speed)
        low_stop = torch.round(torch.sigmoid(out_stop))
        
        # logger.info(f"high-level pred: {action}")
        # logger.info(f"low-level pred: {out_vel} {low_stop}")
        
        self._vel_control.linear_velocity = mn.Vector3(0, 0, lin_vel)
        self._vel_control.angular_velocity = mn.Vector3(0, ang_vel, 0)            
        
        return action.item(), self._vel_control, low_stop
