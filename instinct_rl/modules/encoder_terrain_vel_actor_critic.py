"""EncoderTerrainVelActorCritic: depth encoder + terrain head + proprio velocity estimator.

HIMLoco/DreamWaQ-style: a feed-forward estimator predicts the current base linear velocity (3-dim)
from proprioceptive history and injects the detached estimate into the actor observation, so the
actor consumes an explicit velocity feature that at deploy time is produced from proprioception
alone. The estimate is supervised against the privileged critic velocity by EstimatorWasabiPPO.
The terrain head reconstructs the privileged height map from the depth latent (auxiliary).
"""

import torch

from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import (
    get_subobs_by_components,
    get_subobs_size,
    replace_obs_components,
)


class EncoderTerrainVelActorCritic(EncoderActorCriticMixin, ActorCritic):
    def __init__(
        self,
        obs_format,
        num_actions,
        encoder_configs,
        critic_encoder_configs=None,
        terrain_latent_component="parallel_latent_0_depth_encoder",
        estimator_target_components=None,
        terrain_predictor_configs=None,
        vel_estimator_obs_components=None,
        vel_target_components=None,
        vel_estimator_configs=None,
        **kwargs,
    ):
        if estimator_target_components is None:
            estimator_target_components = []
        if terrain_predictor_configs is None:
            terrain_predictor_configs = {}
        if vel_estimator_obs_components is None:
            vel_estimator_obs_components = []
        if vel_target_components is None:
            vel_target_components = []
        if vel_estimator_configs is None:
            vel_estimator_configs = {}

        super().__init__(
            obs_format=obs_format,
            num_actions=num_actions,
            encoder_configs=encoder_configs,
            critic_encoder_configs=critic_encoder_configs,
            **kwargs,
        )

        self.terrain_latent_component = terrain_latent_component
        self.vel_estimator_obs_components = list(vel_estimator_obs_components)
        self.vel_target_components = list(vel_target_components)
        terrain_target_components = list(estimator_target_components)

        # EstimatorWasabiPPO reads the supervision target from the critic obs in critic-segment
        # key order. Velocity targets precede the terrain target there (base_lin_vel is an early
        # critic term, height_map_priv the last), so estimated_state_ must be concatenated in the
        # same order below. Assert to fail fast if the config order ever diverges.
        combined = self.vel_target_components + terrain_target_components
        critic_ordered = [
            c
            for c in self.critic_obs_segments
            if c in set(self.vel_target_components) | set(terrain_target_components)
        ]
        assert combined == critic_ordered, (
            f"estimator target order {combined} must match critic obs order {critic_ordered}"
        )
        self.estimator_obs_components = None
        self.estimator_target_components = combined

        vel_input_size = get_subobs_size(self.obs_segments, self.vel_estimator_obs_components)
        vel_output_size = get_subobs_size(self.critic_obs_segments, self.vel_target_components)
        assert get_subobs_size(self.obs_segments, self.vel_target_components) == vel_output_size, (
            "velocity target must have equal size in policy obs (injection) and critic obs (supervision)"
        )
        self.vel_estimator = MlpModel(
            input_size=vel_input_size,
            output_size=vel_output_size,
            **vel_estimator_configs,
        )

        latent_dim = self.encoders.output_segment[terrain_latent_component][0]
        terrain_output_size = get_subobs_size(self.critic_obs_segments, terrain_target_components)
        self.state_estimator = MlpModel(
            input_size=latent_dim,
            output_size=terrain_output_size,
            **terrain_predictor_configs,
        )

    def _estimate_velocity(self, observations):
        vel_input = get_subobs_by_components(
            observations, self.vel_estimator_obs_components, self.obs_segments
        )
        return self.vel_estimator(vel_input)

    def act(self, observations, **kwargs):
        observations = observations.clone()
        vel_estimate = self._estimate_velocity(observations)
        observations = replace_obs_components(
            observations, self.vel_target_components, vel_estimate.detach(), self.obs_segments
        )
        obs = self.encoders(observations)
        latent = get_subobs_by_components(
            obs, [self.terrain_latent_component], self.encoders.output_segment
        )
        terrain_estimate = self.state_estimator(latent)
        self.estimated_state_ = torch.cat([vel_estimate, terrain_estimate], dim=-1)
        return ActorCritic.act(self, obs, **kwargs)

    def act_inference(self, observations):
        observations = observations.clone()
        vel_estimate = self._estimate_velocity(observations)
        observations = replace_obs_components(
            observations, self.vel_target_components, vel_estimate.detach(), self.obs_segments
        )
        obs = self.encoders(observations)
        return ActorCritic.act_inference(self, obs)

    def get_estimated_state(self):
        return self.estimated_state_
