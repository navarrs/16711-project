# ------------------------------------------------------------------------------
# @file     seq2seq_policy.py
# @brief    implements a seq2seq policy for a high-level controller
# ------------------------------------------------------------------------------
import abc
import torch
import torch.nn as nn

from controller.models.encoders.simple_cnns import (
    SimpleDepthCNN,
    SimpleRGBCNN,
)
from controller.models.encoders.resnet_encoders import (
    VlnResnetDepthEncoder, 
    TorchVisionResNet50
)
from gym import Space
from controller.models.policies.policy import BasePolicy
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net


class Seq2SeqPolicy(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            Seq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class Seq2SeqNet(Net):
    r"""
    A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.

    Modules:
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
            )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "SimpleRGBCNN",
            "TorchVisionResNet50",
        ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

        if model_config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
            self.rgb_encoder = SimpleRGBCNN(
                observation_space, model_config.RGB_ENCODER.output_size
            )
        elif model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            self.device = (
                torch.device("cuda", model_config.TORCH_GPU_ID)
                if torch.cuda.is_available() and not model_config.use_cpu
                else torch.device("cpu")
            )
            self.rgb_encoder = TorchVisionResNet50(
                observation_space, model_config.RGB_ENCODER.output_size, self.device 
            )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder       
        rnn_input_size = (
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )
        
        if "pointgoal_with_gps_compass" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["pointgoal_with_gps_compass"].shape[0]
            
        if "rel_pointgoal" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["rel_pointgoal"].shape[0]
            
        if "pointgoal" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["pointgoal"].shape[0]

        if "heading" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["heading"].shape[0]
        
        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim
        
        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
                
        x = torch.cat([depth_embedding, rgb_embedding], dim=1)
        
        pointgoal_encoding = torch.zeros(
            [1, 2], dtype=torch.float32, device=self.device)
        heading_encoding = torch.zeros(
            [1, 2], dtype=torch.float32, device=self.device)
        if "pointgoal_with_gps_compass" in observations:
            pointgoal_encoding = observations["pointgoal_with_gps_compass"]
        elif "pointgoal" in observations:
            pointgoal_encoding = observations["pointgoal"]
        
        if "heading" in observations:
            heading_encoding = observations["heading"]
        
        x = torch.cat([x, pointgoal_encoding, heading_encoding], dim=1)
    
        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states