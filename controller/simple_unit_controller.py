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
        self._proximity_threshold = 0.05     # Depth units TODO
        self._turn_threshold = np.radians(8) # Threshold used to determine if straight or turn is better 
        self._goal_radius = 0.25             # How close the agent needs to be to the waypoint
 
        self.build_controller()
 
        self.last_action = None  # Used in obstacle avoidance to avoid turning back and forth
        self._action_queue = []

    def build_controller(self) -> None:
        """
            Sets the controller up
            Sets the controller up - this controller is too basic to need fancy stuff like this
        """
        pass

    def reset(self) -> None:
        """ """
        self.last_action = None 
        self._action_queue = []



    def empty_queue(self):
        """
            Returns the next action in the action queue, 
                along with the number of remaining actions (after)
        """
        next_action = self._action_queue.pop(0)
        return next_action, not(bool(len(self._action_queue)))  # This return statement is colorful!

    def fill_queue(self, observations):
        """
            Uses the observations to fill up the action queue with an appropriate series of actions by 
                FIRST  checking for termination (i.e., reaching the waypoint)
                SECOND checking for obtacles to avoid
                THIRD  following the path (not sure when this would ever arise but it seems necessary for completion)

            No returns, just fills up a queue member variable
        """

        rho, phi = observations[0]["pointgoal_with_gps_compass"]
        
        # FIRST - Check for Termination
        if rho < self._goal_radius:
            self._action_queue.append(HabitatSimActions.STOP) # You've made it!

        
        # SECOND - Look both ways before crossing the street 
        #    (NOTE - Do a 45* turn if there is an obstacle in the way?)
        depth_map = (observations[0]["depth"]).squeeze()  # TODO May need to trim edges, depending on FoV
        close_pixels = np.where(depth_map  < self._proximity_threshold)

        # TODO one environment has the floor labeled as "Close"
        if len(close_pixels[0]) > 50: # If a significant number of pixels indicate an obstacle...
            left_points = sum(close_pixels[1] < (256/2)) # TODO Verify coordinates


            # TODO If I don't step, it gets stuck, but stepping is sketchy
            if left_points > (len(close_pixels[1]) / 2): # Make a 45* right turn + step
                self._action_queue.extend((HabitatSimActions.TURN_RIGHT, HabitatSimActions.TURN_RIGHT, HabitatSimActions.MOVE_FORWARD))

            else:
                self._action_queue.extend((HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_LEFT, HabitatSimActions.MOVE_FORWARD))


        # THIRD - if SOMEHOW there ISNT an obstacle (why would this even be called then...?), follow the golden brick road TODO
        elif np.abs(phi) <= self._turn_threshold:
            self._action_queue.append( HabitatSimActions.MOVE_FORWARD)

        else:
            numTurns = np.abs(int(np.round(phi / np.radians(self._config.SIMULATOR.TURN_ANGLE))))
            if numTurns <= 0:
                print("Error! Somehow numTurns <= 0!")
            if (phi > 0):
                self._action_queue.extend(([HabitatSimActions.TURN_LEFT] * numTurns))
            else:
                self._action_queue.extend(([HabitatSimActions.TURN_RIGHT] * numTurns))


    def get_next_action(self, 
                         observations,
                         deterministic: Optional[bool] = False, 
                         **kwargs) -> int:
        """
             Checks to see if there is already a pending action in the action queue to be taken. 
                If the queue is empty, this fills it up and then takes the first pending action.

            Returns an action and a flag to signify that the queue is/isnt empty
        """

        if not (len(self._action_queue) == 0):
            return self.empty_queue()

        else:
            self.fill_queue(observations)
            return self.empty_queue()


















# BORING OLD STUFF
"""
    def get_shortest_angle(self, phi, gamma) -> float:
        diff = np.abs(phi - gamma)
        diff = diff % (2 * np.pi)
        if diff > np.pi:
            diff = (2 * np.pi) - diff

        return diff

    def avoid_obstacle(self, observations, close_pixels):
        "" Turns in the direction with fewest close pixels
                (Tracks last action to ensure we don't just wiggle back and forth)
        ""
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
        ""
        ""

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


"""