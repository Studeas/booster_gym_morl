import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs, num_groups):
        super().__init__()
        self.num_groups = num_groups # record the dimensions of w_per_group
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs - num_groups + num_privileged_obs, 256), # no need to add 6, num_obs self adaptive
            # torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256), # no need to add 6, num_obs self adaptive
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(
            torch.full((1, num_act), fill_value=-2.0), requires_grad=True
        )

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)
    
    # def act(self, obs):
    #     action_mean = self.actor(obs)
    #     logstd_clamped = torch.clamp(self.logstd, min=-5.0, max=1.0)
    #     action_std = torch.exp(logstd_clamped).expand_as(action_mean)
    #     return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        base_obs = obs[..., :-self.num_groups]
        critic_input = torch.cat((base_obs, privileged_obs), dim=-1)
        # critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
