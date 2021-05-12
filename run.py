# -----------------------------------------------------------------------------#
# @date     april 3, 2021                                                      #
# @brief    controller test                                                 #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import os
import numpy as np
import safety_verify

from controller.common.environments import SimpleRLEnv
from controller.common.utils import resize, observations_to_image
from controller.controller import BaseController, ControllerType
from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions

BLACK_LIST = ["top_down_map", "fog_of_war_mask", "agent_map_coord"]


def get_results(observations, results, action, is_fallback_on, sim):
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

    infos[0]["is_fallback_on"] = is_fallback_on
    return observations, results, dones, infos

def log(n_episodes, results):
    logger.info(f"Metrics for {n_episodes+1} episodes")
    for k, v in results.items():
        if "collision_count" in k:
            logger.info(f"\t-- {k}: {v}")
        logger.info(f"\t-- avg. {k}: {np.asarray(np.mean(v))}")

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

        robotc = config.ROBOT_CONTROL

        # ----------------------------------------------------------------------
        # Blackbox controller
        bb_controller = base.build_controller(ControllerType.BLACKBOX)

        # ----------------------------------------------------------------------
        # Fallback controller
        if robotc.use_fallback:
            fb_controller = base.build_controller(ControllerType.FALLBACK)

            # ------------------------------------------------------------------
            # Safety verification
            verify = safety_verify.Verify(cell_size=0.125)
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
            done = None
            info = None
            episode_id = env.current_episode.episode_id
            actions_taken = 0
            
            is_backup_done = True
            is_stopped = False
            is_valid_map = False
            is_safe = True
            logger.info(f"Running episode={episode_id}")
            while not env.habitat_env.episode_over:
                is_fallback_on = False

                # Compute blackbox controller action
                if is_backup_done:
                    # ----------------------------------------------------------
                    (
                        high_level_action,  # HabitatSimAction (FORWARD, LEFT, RIGHT, STOP)
                        low_level_action,   # VelocityControl (lin. + ang. velocity)
                        low_stop_action,    # VelocityControlStop
                    ) = bb_controller.get_next_action(
                        observations, deterministic=False, done=done
                    )

                    if config.ROBOT_CONTROL.velocity_control:
                        action = low_level_action
                    else:
                        action = high_level_action

                    # 3. Verify safety of reachable set
                    if robotc.use_fallback:
                        is_safe, is_valid_map, top_down_map = verify.verify_safety(
                            info, 6, action, verbose=False)

                # 4. Run fallback
                if not is_safe and robotc.use_fallback:
                    action, is_backup_done = fb_controller.get_next_action(
                        observations)
                    is_fallback_on = True

                # 5. Take a step
                if (
                    low_stop_action == 1.0 or
                    action == HabitatSimActions.STOP or
                    actions_taken > config.ENVIRONMENT.MAX_EPISODE_STEPS
                ):
                    is_stopped = True
                actions_taken += 1
                obs = [env.step(action)]

                # 6. Unroll results
                observations, results, done, info = get_results(
                    obs, results, action, is_fallback_on, env.habitat_env.sim
                )

                if is_valid_map and robotc.safety_verify.add_forecast:
                    info[0]["top_down_map"]["map"] = top_down_map

                frame = observations_to_image(observations[0], info[0])
                frames.append(frame)

            if not is_stopped:
                observations, results, done, info = get_results(
                    obs, results, HabitatSimActions.STOP, is_fallback_on, 
                    env.habitat_env.sim
                )

            if (i+1) % config.LOG_UPDATE == 0:
                log(i+1, results)

            # save episode
            if frames and len(config.VIDEO_OPTION) > 0:
                if robotc.use_fallback:
                    file = f"{i}_episode={episode_id}_with_fallback"
                else:
                    file = f"{i}_episode={episode_id}"
                frames = resize(frames)
                images_to_video(frames, config.VIDEO_DIR, file)

        log(i+1, results)
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
