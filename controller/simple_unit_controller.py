import habitat_sim
import numpy as np

from gym import Space
from habitat import logger
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from typing import Optional



from matplotlib import pyplot as plt  ########################################

class SimpleUnitController():
    def __init__(self, 
                 config: Config,
                 stop_on_error: bool = True,
                 ) -> None:
        """ 
        Simple PID Controller
        Args
        ----
            config: yaml file with config params
            obs_space: observation space. currently uses: DEPTH, RGB, POINTGOALS     
            act_space: action space. currently uses MOVE_FORWARD, TURN_LEFT,
                TURN_RIGHT, STOP
            stop_on_error: unused
        """
        self._config = config 
        self._stop_on_error = stop_on_error
        self._proximity_threshold = 0.05 #25 # Depth units TODO 

        self.build_controller()

        self.last_action = None  # Used in obstacle avoidance to avoid turning back and forth

    def build_controller(self) -> None:
        """
            Sets the controller up
        """
        pass

    def reset(self) -> None:
        """
        """
        pass 

    def get_shortest_angle(self, phi, gamma) -> float:
        diff = np.abs(phi - gamma)
        diff = diff % (2 * np.pi)
        if diff > np.pi:
            diff = (2 * np.pi) - diff

        return diff

    def avoid_obstacle(self, observations, close_pixels):
        """ Turns in the direction with fewest close pixels
                (Tracks last action to ensure we don't just wiggle back and forth)
        """
        # Determine region of points
        left_points = sum(close_pixels[1] < (256/2))
        if self.last_action is None:
            if left_points > (len(close_pixels[1]) / 2): # Obstacle is primarily on the left, go right
                return HabitatSimActions.TURN_RIGHT
            else:
                return HabitatSimActions.TURN_LEFT
        else:
            if self.last_action is HabitatSimActions.TURN_RIGHT:
                return HabitatSimActions.TURN_RIGHT
            else:
                return HabitatSimActions.TURN_LEFT


    def get_next_action(self, 
                         observations,
                         deterministic: Optional[bool] = False, 
                         **kwargs) -> int:
        """
        """

        # Update Variables
        gamma = observations[0]["heading"].item()
        if gamma > np.pi:
            gamma = (2 * np.pi - gamma) # Pick shortest direction to rotate

        rho, phi = observations[0]["pointgoal_with_gps_compass"]
        tau = np.radians(8)
        theta = phi# self.get_shortest_angle(phi, gamma) 


        if rho < 0.25: # If approximately at end goal (arbitrary threshold chosen)
            return HabitatSimActions.STOP

        # Check for obstacles
        # plt.imshow(observations[0]["depth"]); plt.show()

        depth_map = (observations[0]["depth"]).squeeze()  # TODO May need to trim edges, depending on FoV

        close_pixels = np.where(depth_map  < self._proximity_threshold)
        
        if len(close_pixels[0]) > 60: # If a significant number of pixels indicate an obstacle...
            action = self.avoid_obstacle(observations, close_pixels)
            self.last_action = action
            return action
        # else:
        #     self.last_action = None  # Only track if we are avoiding an obstacle...?


        # If no obstacle, take action

        if (theta > tau) and not (self.last_action ==  HabitatSimActions.TURN_RIGHT):
            self.last_action = HabitatSimActions.TURN_LEFT
            return HabitatSimActions.TURN_LEFT
        elif (theta < -tau) and not (self.last_action == HabitatSimActions.TURN_LEFT):
            self.last_action = HabitatSimActions.TURN_RIGHT
            return HabitatSimActions.TURN_RIGHT

        else:# (np.abs(theta) < tau):
            self.last_action = HabitatSimActions.MOVE_FORWARD
            return HabitatSimActions.MOVE_FORWARD


