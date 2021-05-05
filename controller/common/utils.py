from typing import Dict
import torch
import torch.nn as nn
import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision, append_text_to_image

cv2 = try_cv2_import()

def draw_mask(
    view: np.ndarray, alpha: float = 0.4, color: np.ndarray = np.array([255, 0, 0])) -> np.ndarray:
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * color + (1.0 - alpha) * view)[mask]
    return view

def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    text = "state: "
    
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)
        text += " collision "
    else:
        text += " no-collision "
        
    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
            
        if "fallback_takeover" in info:
            if info["fallback_takeover"]:
                top_down_map = draw_mask(top_down_map, 
                                         color=np.array([0, 255, 0]))
                text += " controller=fallback "
            else:
                top_down_map = draw_mask(top_down_map, 
                                         color=np.array([0, 0, 255]))
                text += " controller=black-box "
           
            
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    
    
    # if "top_down_map_verified" in info:
    #     top_down_map = info["top_down_map_verified"]
    #     top_down_map = maps.colorize_topdown_map(
    #         top_down_map, info["top_down_map"]["fog_of_war_mask"]
    #     )
    #     map_agent_pos = info["top_down_map"]["agent_map_coord"]
    #     top_down_map = maps.draw_agent(
    #         image=top_down_map,
    #         agent_center_coord=map_agent_pos,
    #         agent_rotation=info["top_down_map"]["agent_angle"],
    #         agent_radius_px=top_down_map.shape[0] // 16,
    #     )

    #     if top_down_map.shape[0] > top_down_map.shape[1]:
    #         top_down_map = np.rot90(top_down_map, 1)

    #     # scale top down map to align with rgb view
    #     old_h, old_w, _ = top_down_map.shape
    #     top_down_height = observation_size
    #     top_down_width = int(float(top_down_height) / old_h * old_w)
    #     # cv2 resize (dsize is width first)
    #     top_down_map = cv2.resize(
    #         top_down_map,
    #         (top_down_width, top_down_height),
    #         interpolation=cv2.INTER_CUBIC,
    #     )
    #     frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    frame = append_text_to_image(frame, text)
    return frame


def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map


def resize(images):
    new_images = []
    size = images[0].shape[:2]
    for image in images:
        nimage = cv2.resize(image, (size[1], size[0]))
        new_images.append(nimage)
    return new_images


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
