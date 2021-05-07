import habitat_sim
import numpy as np

from gym import Space
from habitat import logger
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from typing import Optional



from matplotlib import pyplot as plt  ########################################


# I think my goal radius should be much smaller 
class AvoiderAndFollower():
    """
        Controller that greatly prioritizes safety and will avoid obstacles at all costs.
            Secondary purpose is to head towards the target waypoint
    """
    def __init__(self, 
                 config: Config,
                 stop_on_error: bool = True,
                 ) -> None:
        """ 
       
        """
        self._config = config 
        self._stop_on_error = stop_on_error
        
        self._proximity_threshold = 0.1449     # Depth units 0.05
        self._pixel_count_threshold = 50
        self._turn_threshold = np.radians(8) # Threshold used to determine if straight or turn is better 
        self._goal_radius = self._config.TASK.SUCCESS_DISTANCE  # How close the agent needs to be to the waypoint
        self._turns_per_circle = 360 / self._config.SIMULATOR.TURN_ANGLE

        # Non-Constants (Mutables?)
        self._turn_direction = None
        self._num_sequential_turns = 0
        self._num_sequential_circles = 0
        self._adjusted_proximity_threshold = self._proximity_threshold # Used to prevent endless spirals

        self._is_avoiding = False
 
        self.build_controller()
 
    def build_controller(self) -> None:
        """
            Sets the controller up - this controller is too basic to need fancy stuff like this
        """
        pass

    def reset(self) -> None:
        """ """
        self._turn_direction = None 
        self._num_sequential_turns = 0
        self._num_sequential_circles = 0
        self._adjusted_proximity_threshold = self._proximity_threshold
        self._is_avoiding = False

    def step_towards_goal(self, observations) -> int:
        """ Aligns the vehicle with the direction to waypoint, or takes a step if already aligned """
        rho, phi = observations[0]["pointgoal_with_gps_compass"]

        if phi > self._turn_threshold:
            return HabitatSimActions.TURN_LEFT
        elif phi < -self._turn_threshold:
            return HabitatSimActions.TURN_RIGHT
        else:
            return HabitatSimActions.MOVE_FORWARD

    def determine_turn_direction(self, observations) -> bool:
        """
            Determines which direction to turn in in order to avoid the obstacle
                (Currently turns in the direction which has less obstacle)
              Once direction has been established, checks to see if agent needs to continue turning or is safe      
        """

        # Check if the obstacle has been avoided
        depth_map = (observations[0]["depth"]).squeeze()  # TODO May need to trim edges, depending on FoV
        
        # Squeeze edges to avoid the whole floor thing
        depth_map = depth_map[:200,:]

        close_pixels = (depth_map < self._adjusted_proximity_threshold) & (depth_map > 0) # Need to avoid counting the gaping pit into the abyss


        if (np.sum(close_pixels)) <= self._pixel_count_threshold: # No obstacle. Yay!
            if self._is_avoiding:  # If we were running an avoidance sequence, finish with a forward step to avoid endless jitter
                is_safe = True
                self.reset()
                next_action = HabitatSimActions.MOVE_FORWARD # Step to prevent endless jitter
                self._is_avoiding = False
            else: # Weren't trying to avoid anything. NOTE This is the ONLY step that actually tries to navigate to a waypoint
                next_action = self.step_towards_goal(observations)
                is_safe = True 
                self._is_avoiding = False

        else: 
            # If the turn direction has not yet been established, determine if you should be turning left or right
            if (self._turn_direction == None): 
                left_points = np.sum(close_pixels[:, :int(256/2)]) # How many close pixels are on the left side of the screen
                
                if left_points > (np.sum(close_pixels) / 2.0):
                    self._turn_direction = HabitatSimActions.TURN_RIGHT
                else:
                    self._turn_direction = HabitatSimActions.TURN_LEFT

            is_safe = False 
            next_action = self._turn_direction
            self._num_sequential_turns += 1
            self._is_avoiding = True

        
            if self._num_sequential_turns >= self._turns_per_circle:
                self._num_sequential_circles += 1
                self._num_sequential_turns = 0
                self._adjusted_proximity_threshold *= 0.8  # If you've gone a whole circle, lower expectations and repeat (like when youre hungry)

            
        return next_action, is_safe
        
    def get_next_action(self, 
                         observations,
                         deterministic: Optional[bool] = False, 
                         **kwargs) -> int:

        # CHECK FOR OBSTACLE 
        rho, phi = observations[0]["pointgoal_with_gps_compass"]
        
        if rho < self._goal_radius:
            next_action = HabitatSimActions.STOP # You did it. Hooray!
            is_safe = True
        
        else:
            next_action, is_safe = self.determine_turn_direction(observations)

        return next_action, is_safe


