#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from habitat_sim.registry import registry
from habitat_sim.sensor import SensorType
from habitat_sim.sensors.noise_models.sensor_noise_model import SensorNoiseModel

try:
    import torch
except ImportError:
    torch = None


@registry.register_noise_model(name="None")
class NoSensorNoiseModel(SensorNoiseModel):
    @staticmethod
    def is_valid_sensor_type(sensor_type: SensorType) -> bool:
        return True

    def apply(self, x):
        if isinstance(x, np.ndarray):
            return x.copy()
        elif torch is not None and torch.is_tensor(x):
            return x.clone()
        else:
            return x
