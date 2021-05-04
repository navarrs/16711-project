# -----------------------------------------------------------------------------#
# @date     april 3, 2021                                                      #
# @brief    controller test                                                 #
# -----------------------------------------------------------------------------#
import habitat_sim
import math
import magnum as mn
import numpy as np
import os
import random
from habitat_sim.utils import viz_utils as vut

dir_path = ""
data_path = os.path.join(dir_path, "data")
output_path = os.path.join(dir_path, "out/")

def make_configuration():
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene.id = os.path.join(
        data_path, "habitat-test-scenes/apartment_1.glb"
    )
    assert os.path.exists(backend_cfg.scene.id)
    backend_cfg.enable_physics = True

    # sensor configurations
    # Note: all sensors must have the same resolution
    # setup 2 rgb sensors for 1st and 3rd person views
    camera_resolution = [544, 720]
    sensors = {
        "rgba_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_camera_1stperson": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": camera_resolution,
            "position": [0.0, 0.6, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        },
        "rgba_camera_3rdperson": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": camera_resolution,
            "position": [0.0, 1.0, 0.3],
            "orientation": [-45, 0.0, 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.SensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.orientation = sensor_params["orientation"]
        sensor_specs.append(sensor_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


class Sim():
    def __init__(self, config: str, episode_config: str = None):
        self._sim = habitat_sim.Simulator(config)
        self._episode = {
            "agent_position": [-0.15, -0.7, 1.0],
            "agent_rotation": np.quaternion(-0.83147, 0, 0.55557, 0),
            "index": 0,
            "agent": str(os.path.join(data_path, "objects/locobot_merged"))
        }

        # setup stuff
        self._object_manager = self._sim.get_object_template_manager()
        
        self.place_agent()
    
    def place_agent(self):
        # place our agent in the scene
        agent_state = habitat_sim.AgentState()
        agent_state.position = self._episode["agent_position"]
        agent_state.rotation = self._episode["agent_rotation"]
        agent = self._sim.initialize_agent(self._episode["index"], agent_state)
        agent.scene_node.transformation_matrix()
        
        robot_object_template = self._object_manager.load_object_configs(
            self._episode["agent"])[0]
        self._robot_id = self._sim.add_object(
            robot_object_template, self._sim.agents[0].scene_node)
        self._sim.set_translation(
            np.array([1.75, -1.02, 0.4]), self._robot_id)
        self._velocity_control = self._sim.get_object_velocity_control(
            self._robot_id)

    def velocity_control(
        self, 
        linear = [0.0, 0.0, 0.0],
        angular = [0.0, 0.0, 0.0], 
        **kwargs
    ):
        
        controlling_lin_vel = kwargs.get('controlling_lin_vel', None)
        if not controlling_lin_vel is None:
            self._velocity_control.controlling_lin_vel = controlling_lin_vel
        
        controlling_ang_vel = kwargs.get('controlling_ang_vel', None)
        if not controlling_ang_vel is None:
            self._velocity_control.controlling_ang_vel = controlling_ang_vel
            
        lin_vel_is_local = kwargs.get('lin_vel_is_local', None)
        if not lin_vel_is_local is None:
            self._velocity_control.lin_vel_is_local = lin_vel_is_local
        
        ang_vel_is_local = kwargs.get('ang_vel_is_local', None)
        if not ang_vel_is_local is None:
            self._velocity_control.ang_vel_is_local = ang_vel_is_local
         
        self._velocity_control.linear_velocity = np.asarray(linear)
        self._velocity_control.angular_velocity = np.asarray(angular)
        
        
    def step(self, dt: float = 1.0):
        observations = []
        start_time = self._sim.get_world_time()
        while self._sim.get_world_time() < start_time + dt:
            self._sim.step_physics(1.0/60.0)
            observations.append(self._sim.get_sensor_observations())
        return observations
    
    def close(self):
        self._sim.remove_object(self._robot_id, False)
            
config = make_configuration()
sim = Sim(config)

sim.velocity_control(
    linear=[0, 0, 0.0], angular=[0.0, 0.0, 0])
observations = sim.step(dt=1.5)

sim.velocity_control(
    linear=[0, 0, -1.5], angular=[0.0, 0.0, 2.0], 
    controlling_lin_vel = True, controlling_ang_vel = True,
    lin_vel_is_local = True, ang_vel_is_local = True)
observations += sim.step(dt=1.0)

sim.velocity_control(
    linear=[0, 0, 1.5], angular=[0.0, 3.0, 0], controlling_lin_vel = False)
observations += sim.step(dt=1.5)

sim.velocity_control(
    linear=[0, 0, 0.0], angular=[0.0, 0.0, 0], 
    controlling_lin_vel = True, controlling_ang_vel = True)
observations += sim.step(dt=1.0)

sim.velocity_control(
    linear=[0, 0, 0.0], angular=[0.0, -1.25, 0], 
    controlling_lin_vel = True, controlling_ang_vel = True)
observations += sim.step(dt=2.0)

vut.make_video(
    observations, "rgba_camera_1stperson", "color",
    output_path + "robot_control",
    open_vid=True)