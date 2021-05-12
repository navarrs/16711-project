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


def get_results(
    observations, results, action, steps, is_fallback_done, is_forecast_on,
    top_down_map, sim
):
    observations, _, dones, infos = [list(x) for x in zip(*observations)]

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

    infos[0]["is_fallback_on"] = not is_fallback_done
    if is_forecast_on:
        infos[0]["top_down_map"]["map"] = top_down_map

    infos[0]["steps_taken"] = steps
    frame = observations_to_image(observations[0], infos[0])
    return observations, results, dones, infos, frame


def log(n_episodes: int, results: dict) -> None:
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
        # ----------------------------------------------------------------------
        # Initialize parameters
        motion_control = config.MOTION_CONTROL
        base = BaseController(
            config, obs_space=env.observation_space, act_space=env.action_space,
            sim=env.habitat_env.sim
        )
        results = {}
        results["collision_distance"] = []

        # ----------------------------------------------------------------------
        # Blackbox controller
        bb_controller = base.build_controller(ControllerType.BLACKBOX)

        # ----------------------------------------------------------------------
        # Fallback controller
        if motion_control.use_fallback:
            fb_controller = base.build_controller(ControllerType.FALLBACK)

            # ------------------------------------------------------------------
            # Safety verification
            verify = safety_verify.Verify(
                cell_size=motion_control.SAFETY_VERIFY.cell_size)
            
            velocities = np.array(motion_control.SAFETY_VERIFY.velocities)
            steers = motion_control.SAFETY_VERIFY.steers
            steers = np.linspace(steers[0], steers[1], steers[2])
            T = motion_control.SAFETY_VERIFY.T
            dt = motion_control.SAFETY_VERIFY.dt
            
            verify.gen_primitive_lib(velocities, steers, dt)

        # ----------------------------------------------------------------------
        # Running episodes
        for i, episode in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break

            # Reset parameters
            frames = []
            observations = [env.reset()]
            bb_controller.reset()
            done = None
            info = None
            top_down_map = None
            episode_id = env.current_episode.episode_id
            actions_taken = 0
            is_fallback_done = True
            is_stopped = False
            is_valid_map = False
            is_safe = True
            logger.info(f"Running episode={episode_id}")

            while not env.habitat_env.episode_over:

                # Compute blackbox controller action
                if is_fallback_done:
                    # Let blackbox take an action:
                    #   High-Level action -> FORWARD, LEFT, RIGHT, STOP
                    #   Low-Level action  -> [LINEAR, ANGULAR]_VELOCITY
                    #   Low-Level stopping decision
                    (
                        high_level_action, low_level_action, low_level_stop
                    ) = bb_controller.get_next_action(
                        observations, deterministic=False, done=done)

                    # Depending on the type of control, the environment will
                    # take a "discrete" or "continous" action
                    if config.MOTION_CONTROL.velocity_control:
                        action = low_level_action
                    else:
                        action = high_level_action

                    # If we're using the fallback, check the safety of the step
                    # before taking any steps
                    if motion_control.use_fallback:
                        (
                            is_safe, is_valid_map, top_down_map
                        ) = verify.verify_safety(info, T, action, verbose=False)

                # Trigger fallback
                if not is_safe and motion_control.use_fallback:
                    action, is_fallback_done = fb_controller.get_next_action(
                        observations)

                # Take a step
                is_stopped = (
                    action == HabitatSimActions.STOP or
                    actions_taken > config.ENVIRONMENT.MAX_EPISODE_STEPS
                )
                actions_taken += 1
                obs = [env.step(action)]

                # Unroll observations
                observations, results, done, info, frame = get_results(
                    obs, results, action, actions_taken, is_fallback_done,
                    (is_valid_map and motion_control.SAFETY_VERIFY.add_forecast),
                    top_down_map, env.habitat_env.sim
                )      
                frames.append(frame)

            # In case the environment stops without saving the last step
            if not is_stopped:
                action = HabitatSimActions.STOP
                observations, results, done, info, frame = get_results(
                    obs, results, action, actions_taken, is_fallback_done,
                    (is_valid_map and motion_control.SAFETY_VERIFY.add_forecast),
                    top_down_map, env.habitat_env.sim
                )      
                frames.append(frame)

            # Save episode
            if (i+1) % config.LOG_UPDATE == 0:
                log(i+1, results)

            if frames and len(config.VIDEO_OPTION) > 0:
                file = f"{i}_episode={episode_id}"
                frames = resize(frames)
                images_to_video(frames, config.VIDEO_DIR, file)

        # Final log and close
        log(i+1, results)
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config", type=str, required=True,
        help="path to config yaml containing info about experiment",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()
