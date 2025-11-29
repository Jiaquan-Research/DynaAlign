import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    """ Simple MLP actor-critic. """
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOAgent:
    """
    Minimal PPO agent used in V9.3 experiments.
    Purposefully simple for full reproducibility.
    """

    def __init__(self, obs_dim, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2):
        self.gamma = gamma
        self.lam = lam
        self.clip = clip

        self.model = ActorCritic(obs_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mu, _ = self.model(obs_t)
        action = mu.item()
        return action

    def update(self, obs, actions, rewards, values, dones):
        """
        Standard PPO update with GAE.
        obs, actions, rewards, values, dones are numpy arrays.
        """

        # convert to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

        # compute advantages (GAE)
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (0 if t == len(rewards) - 1 else values[t + 1]) - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)

        # normalize advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # old policy
        mu_old, _ = self.model(obs)
        ratio = (mu_old - actions).abs()  # simplified surrogate for 1-D control

        # clipped objective
        loss_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
        loss = -torch.min(loss_clip, ratio * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
