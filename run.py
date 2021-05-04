# -----------------------------------------------------------------------------#
# @date     april 3, 2021                                                      #
# @brief    controller test                                                 #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import habitat_sim
import os
import magnum as mn
import numpy as np
import torch

from controller.common.environments import SimpleRLEnv
from controller.common.utils import resize, observations_to_image
from controller.controller import BaseController, ControllerType
from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def run_exp(exp_config: str) -> None:
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)

    if not os.path.exists(config.VIDEO_DIR) and len(config.VIDEO_OPTION) > 0:
        os.makedirs(config.VIDEO_DIR)

    with SimpleRLEnv(config=config) as env:
        base = BaseController(config, 
                              obs_space=env.observation_space, 
                              act_space=env.action_space,
                              sim=env.habitat_env.sim)

        # ----------------------------------------------------------------------
        # Blackbox controller
        bb_controller = base.build_controller(ControllerType.BLACKBOX)

        # ----------------------------------------------------------------------
        # Fallback controller
        # @TODO: add fallback controller
        # fb_controller = base.build_controller(ControllerType.FALLBACK)

        vc = habitat_sim.physics.VelocityControl()
        if config.ROBOT.velocity_control:
            vc.controlling_lin_vel = True
            vc.lin_vel_is_local = True
            vc.controlling_ang_vel = True
            vc.ang_vel_is_local = True
            vc.linear_velocity = np.array([0, 0, 1.0])
        
        # ----------------------------------------------------------------------
        for i, episode in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break
            
            frames = []
            observations = [env.reset()]
            bb_controller.reset()
            dones = None
            goal_pos = env.current_episode.goals[0].position
            
            while not env.habitat_env.episode_over:

                # 1. Compute blackbox controller action
                action = bb_controller.get_next_action(
                    observations, deterministic=True, dones=dones, goal_pos=goal_pos)
                
                # @TODO: convert this to vel_control 
                
                # 2. @TODO: Compute future estimates

                # 3. @TODO: Verify safety of reachable set
                safe = True
                if not safe:
                    # 4. @TODO: Compute fallback controller action
                    action = None

                # 5. Take a step
                observations = [env.step(action)]
                observations, rewards, dones, infos = [
                    list(x) for x in zip(*observations)
                ]

                frame = observations_to_image(observations[0], infos[0])
                frames.append(frame)

            # save episode
            if frames and len(config.VIDEO_OPTION) > 0:
                frames = resize(frames)
                images_to_video(
                    frames,
                    config.VIDEO_DIR,
                    "path_id={}={}".format(
                        episode.episode_id, i
                    )
                )

        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()
