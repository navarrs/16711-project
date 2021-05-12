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
import safety_verify

from controller.common.environments import SimpleRLEnv
from controller.common.utils import resize, observations_to_image
from controller.controller import BaseController, ControllerType
from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.utils import generate_video

BLACK_LIST = ["top_down_map", "fog_of_war_mask", "agent_map_coord"]


def unroll_results(observations, results, action):
    observations, rewards, dones, infos = [
        list(x) for x in zip(*observations)
    ]

    if action == HabitatSimActions.STOP:
        for k, v in infos[0].items():
            if k in BLACK_LIST:
                continue
            if "collisions" in k:
                if results.get("collision_count") == None:
                    results["collision_count"] = []
                results["collision_count"].append(v["count"])
            else:
                if results.get(k) == None:
                    results[k] = []
                results[k].append(v)
    return observations, results, dones, infos


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
        
        # ----------------------------------------------------------------------
        # Safety verification
        # verify = safety_verify.Verify(cell_size=.125)
        # verify.gen_primitive_lib(
        #     np.array([.25]), np.linspace(-np.pi/12, np.pi/12, 3))

        # # ----------------------------------------------------------------------
        # # Define desired parameters
        # vc_desired = habitat_sim.physics.VelocityControl()
        # vc_desired.controlling_lin_vel = True
        # vc_desired.lin_vel_is_local = True
        # vc_desired.controlling_ang_vel = True
        # vc_desired.ang_vel_is_local = True

        # # @TODO: change these parameters ?
        # vc_desired.linear_velocity = np.array([0, 0, -1.0])
        # vc_desired.angular_velocity = np.array([0.0, 0.0, 0.0])

        # note: this is the velocity we will feed to the environment
        # vc_current = vc_desired

        # ----------------------------------------------------------------------
        results = {}
        for i, episode in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break

            frames = []
            observations = [env.reset()]
            bb_controller.reset()
            dones = None
            infos = None
            goal_pos = env.current_episode.goals[0].position
            scene_id = env.current_episode.scene_id
            episode_id = env.current_episode.episode_id

            if "van-gogh" in scene_id:
                continue
                
            max_actions = 1000
            actions_taken = 0
            while not env.habitat_env.episode_over:

                # Compute blackbox controller action
                # --------------------------------------------------------------
                action, vc, low_stop = bb_controller.get_next_action(
                    observations, deterministic=True)
                
        #         # @TODO: convert the action to vel_control
        #         # --------------------------------------------------------------
        #         vc_bb = habitat_sim.physics.VelocityControl()

        #         # Verify safety of reachable set
        #         # --------------------------------------------------------------
        #         safe = verify.verify_safety(infos, 6, action, verbose=False)
        #         if not safe and config.CONTROLLERS.use_fallback:
        #             # @TODO: Compute fallback controller action
        #             # action = fb_controller.get_next_action(observations)
        #             vc_fb = fb_controller.get_next_action(observations)

                if action == HabitatSimActions.STOP or actions_taken > max_actions:
                    break
                actions_taken +=1
                
                # Take a step
                observations = [env.step(vc)]
                observations, results, dones, infos = unroll_results(
                    observations, results, action)
                frame = observations_to_image(observations[0], infos[0])
                frames.append(frame)
                
            # if (i+1) % config.LOG_UPDATE == 0:
            #     logger.info(f"Metrics for {i+1} episodes")
            #     for k, v in results.items():
            #         logger.info(f"\t -- avg. {k}: {np.asarray(np.mean(v))}")

            # # save episode
            if frames and len(config.VIDEO_OPTION) > 0:
                frames = resize(frames)
                time_step = 30
                images_to_video(
                    frames,
                    config.VIDEO_DIR,
                    "path_id={}={}".format(
                        episode.episode_id, i
                    ),
                    fps=int(1.0/time_step)
                )
               
                
                
        logger.info(f"Done. Metrics for {i+1} episodes")
        for k, v in results.items():
            logger.info(f"\t -- avg. {k}: {np.asarray(np.mean(v))}")

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
