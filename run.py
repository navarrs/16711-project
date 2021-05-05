# -----------------------------------------------------------------------------#
# @date     april 3, 2021                                                      #
# @brief    controller test                                                 #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import os
import torch
import numpy as np
import safety_verify

from controller.common.environments import SimpleRLEnv
from controller.common.utils import resize, observations_to_image
from controller.controller import BaseController, ControllerType
from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from matplotlib import pyplot as plt

BLACK_LIST = ["top_down_map", "fog_of_war_mask", "agent_map_coord"]


def get_results(observations, results, action, fallback_takeover, sim):
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
    if results.get("collision_distance") == None:
        results["collision_disance"] = []
    
    collision_distance = sim.distance_to_closest_obstacle(
        sim.get_agent_state().position, 1.0)
    results["collision_distance"].append(collision_distance)

    infos[0]["fallback_takeover"] = fallback_takeover
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
        fb_controller = base.build_controller(ControllerType.FALLBACK)

        # ----------------------------------------------------------------------
        # Safety verification
        verify = safety_verify.Verify(cell_size=.125)
        verify.gen_primitive_lib(
            np.array([.25]), np.linspace(-np.pi/12, np.pi/12, 3))

        # ----------------------------------------------------------------------
        # Running episodes
        results = {}
        results["collision_distance"] = []
        for i, episode in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break

            frames = []
            observations = [env.reset()]
            bb_controller.reset()
            infos = None
            dones = None
            goal_pos = env.current_episode.goals[0].position
            scene_id = env.current_episode.scene_id
            episode_id = env.current_episode.episode_id

            backup_is_done = True

            while not env.habitat_env.episode_over:
                fallback_takeover = False

                if backup_is_done:
                    # 1. Compute blackbox controller action
                    action = bb_controller.get_next_action(
                        observations, deterministic=False, dones=dones, 
                        goal_pos=goal_pos)

                    # 2. @TODO: Compute future estimates

                    # 3. Verify safety of reachable set
                    safe, is_valid_map, top_down_map = verify.verify_safety(
                        infos, 6, action, verbose=False)

                # safe = False
                if not safe and config.CONTROLLERS.use_fallback:
                    # 4. Compute fallback controller action
                    action, backup_is_done = fb_controller.get_next_action(
                        observations)
                    fallback_takeover = True

                # 5. Take a step
                observations = [env.step(action)]

                # 6. Unroll results
                observations, results, dones, infos = get_results(
                    observations, 
                    results, 
                    action, 
                    fallback_takeover,
                    env.habitat_env.sim
                )

                if is_valid_map:
                    infos[0]["top_down_map"]["map"] = top_down_map

                frame = observations_to_image(observations[0], infos[0])
                frames.append(frame)

            if (i+1) % config.LOG_UPDATE == 0:
                logger.info(f"Metrics for {i+1} episodes")
                for k, v in results.items():
                    logger.info(f"\t -- avg. {k}: {np.asarray(np.mean(v))}")

            # save episode
            if frames and len(config.VIDEO_OPTION) > 0:
                if config.CONTROLLERS.use_fallback:
                    file = f"{i}_episode_id={episode_id}_with_fallback"
                else:
                    file = f"{i}_episode_id={episode_id}"

                frames = resize(frames)
                images_to_video(frames, config.VIDEO_DIR, file)

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
