"""Encoder actor-critic with a velocity-only proprio estimator."""

from __future__ import annotations

import torch

from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size, replace_obs_components


class EncoderVelActorCritic(EncoderActorCriticMixin, ActorCritic):
    """Depth encoder + MLP base-velocity estimator injected into the actor obs."""

    def __init__(
        self,
        obs_format,
        num_actions,
        encoder_configs,
        critic_encoder_configs=None,
        vel_estimator_obs_components=None,
        vel_target_components=None,
        vel_estimator_configs=None,
        **kwargs,
    ):
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

        self.vel_estimator_obs_components = list(vel_estimator_obs_components)
        self.vel_target_components = list(vel_target_components)
        self.estimator_obs_components = self.vel_estimator_obs_components
        self.estimator_target_components = self.vel_target_components

        vel_input_size = get_subobs_size(self.obs_segments, self.vel_estimator_obs_components)
        vel_output_size = get_subobs_size(self.critic_obs_segments, self.vel_target_components)
        policy_target_size = get_subobs_size(self.obs_segments, self.vel_target_components)
        assert policy_target_size == vel_output_size, (
            "velocity target components must have equal size in policy obs "
            f"({policy_target_size}) and critic obs ({vel_output_size})."
        )

        self.vel_estimator = MlpModel(
            input_size=vel_input_size,
            output_size=vel_output_size,
            **vel_estimator_configs,
        )
        self.estimated_state_ = torch.zeros(1, vel_output_size)

    def _estimate_velocity(self, observations):
        vel_input = get_subobs_by_components(
            observations,
            self.vel_estimator_obs_components,
            self.obs_segments,
        )
        return self.vel_estimator(vel_input)

    def _inject_velocity(self, observations, vel_estimate):
        return replace_obs_components(
            observations,
            self.vel_target_components,
            vel_estimate.detach(),
            self.obs_segments,
        )

    def act(self, observations, **kwargs):
        observations = observations.clone()
        vel_estimate = self._estimate_velocity(observations)
        observations = self._inject_velocity(observations, vel_estimate)
        obs = self.encoders(observations)
        self.estimated_state_ = vel_estimate
        return ActorCritic.act(self, obs, **kwargs)

    def act_inference(self, observations):
        observations = observations.clone()
        vel_estimate = self._estimate_velocity(observations)
        observations = self._inject_velocity(observations, vel_estimate)
        obs = self.encoders(observations)
        return ActorCritic.act_inference(self, obs)

    def get_estimated_state(self):
        return self.estimated_state_
