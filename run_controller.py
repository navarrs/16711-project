# -----------------------------------------------------------------------------#
# @date     April 3, 2021                                                      #
# @brief    rl controller test                                                 #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import os
import torch

from controller.common.environments import SimpleRLEnv
from controller.common.utils import resize, observations_to_image
from controller.ppo_controller import PPOController
from habitat import logger
from habitat.utils.visualizations.utils import images_to_video
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.utils.common import batch_obs


def run_controller(exp_config: str) -> None:
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)

    if not os.path.exists(config.VIDEO_DIR) and len(config.VIDEO_OPTION) > 0:
        os.makedirs(config.VIDEO_DIR)

    with SimpleRLEnv(config=config) as env:

        rl_controller = PPOController(
            config=config,
            obs_space=env.observation_space,
            act_space=env.action_space
        )
        device = rl_controller.get_device()

        for i, episode in enumerate(env.episodes):

            frames = []
            observations = [env.reset()]

            test_recurrent_hidden_states = torch.zeros(
                rl_controller.get_actor_critic().net.num_recurrent_layers,
                config.NUM_PROCESSES,
                config.RL.PPO.hidden_size,
                device=device
            )
            prev_action = torch.zeros(
                config.NUM_PROCESSES, 1, device=device, dtype=torch.long)

            not_done_masks = torch.zeros(
                config.NUM_PROCESSES, 1, device=device)

            while not env.habitat_env.episode_over:
                batch = batch_obs(observations, device=device)

                (
                    action,
                    test_recurrent_hidden_states
                ) = rl_controller.get_next_action(
                    batch,
                    test_recurrent_hidden_states,
                    prev_action,
                    not_done_masks,
                    True  # this is going to sample actions
                )

                action = action.item()
                observations = [env.step(action)]
                observations, rewards, dones, infos = [
                    list(x) for x in zip(*observations)
                ]
                frame = observations_to_image(observations[0], infos[0])
                frames.append(frame)

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=device,
                )
            
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
    run_controller(**vars(args))


if __name__ == "__main__":
    main()
