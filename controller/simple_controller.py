# ------------------------------------------------------------------------------
# @brief    wrapper for a PPO-based controller
# ------------------------------------------------------------------------------
import habitat_sim
import numpy as np

from gym import Space
from habitat import logger
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from typing import Optional

class SimpleController():
    def __init__(
        self, config: Config,
        sim: HabitatSim,
        goal_radius: float = 0.5,
        stop_on_error: bool = True,
    ) -> None:
        """
        Simple controller
        Args
        ----
            config: yaml file with config params
            sim: simulation enviroment. used to create a greedy controller 
        """
        self._config = config
        self._sim = sim
        self._goal_radius = goal_radius
        self._stop_on_error = stop_on_error

        self.build_controller()

    def build_controller(self) -> None:
        """
        Sets the controller up
        """
        self._controller = self._sim._sim.make_greedy_follower(
            0,
            self._goal_radius,
            stop_key=HabitatSimActions.STOP,
            forward_key=HabitatSimActions.MOVE_FORWARD,
            left_key=HabitatSimActions.TURN_LEFT,
            right_key=HabitatSimActions.TURN_RIGHT,
        )

    def reset(self) -> None:
        pass

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
        goal_pos = kwargs.get('goal_pos', None)
        if goal_pos is None:
            return HabitatSimActions.STOP

        try:
            next_action = self._controller.next_action_along(goal_pos)
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                next_action = HabitatSimActions.STOP
            else:
                raise e

        return next_action
