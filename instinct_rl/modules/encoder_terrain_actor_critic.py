from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class EncoderTerrainActorCritic(EncoderActorCriticMixin, ActorCritic):
    """Encoder actor-critic with terrain height-map prediction from perceptive latent."""

    def __init__(
        self,
        obs_format,
        num_actions,
        encoder_configs,
        critic_encoder_configs=None,
        terrain_latent_component="parallel_latent_0_perceptive",
        estimator_target_components=None,
        terrain_predictor_configs=None,
        **kwargs,
    ):
        if estimator_target_components is None:
            estimator_target_components = []
        if terrain_predictor_configs is None:
            terrain_predictor_configs = {}
        super().__init__(
            obs_format=obs_format,
            num_actions=num_actions,
            encoder_configs=encoder_configs,
            critic_encoder_configs=critic_encoder_configs,
            **kwargs,
        )
        self.terrain_latent_component = terrain_latent_component
        self.estimator_obs_components = None
        self.estimator_target_components = estimator_target_components

        latent_dim = self.encoders.output_segment[terrain_latent_component][0]
        target_dim = get_subobs_size(self.critic_obs_segments, estimator_target_components)
        self.state_estimator = MlpModel(
            input_size=latent_dim,
            output_size=target_dim,
            **terrain_predictor_configs,
        )

    def act(self, observations, **kwargs):
        obs = self.encoders(observations)
        latent = get_subobs_by_components(
            obs,
            [self.terrain_latent_component],
            self.encoders.output_segment,
        )
        self.estimated_state_ = self.state_estimator(latent)
        return ActorCritic.act(self, obs, **kwargs)

    def act_inference(self, observations):
        obs = self.encoders(observations)
        return ActorCritic.act_inference(self, obs)

    def get_estimated_state(self):
        return self.estimated_state_
