# ============================================================
# Standard PPO Agent Implementation for DynaAlign
# Core Policy Network & Proximal Policy Optimization Logic
# (Kept consistent with V8.3 / V9.3 training runs)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters (keep original values!)
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 6
HIDDEN = 64
LR = 1e-3  # High LR to capture system sensitivity


# ============================================================
# Actor-Critic Network
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(3, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )

        # Policy head
        self.mu_head = nn.Linear(HIDDEN, 1)
        # Keep original initialization
        self.log_std = nn.Parameter(torch.tensor(-1.0))

        # Value head
        self.v_head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std)
        v = self.v_head(h).squeeze(-1)
        return mu, std, v


# ============================================================
# PPO Agent
# ============================================================
class PPOAgent:
    def __init__(self, lr: float = LR):
        self.net = ActorCritic().to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.epochs = PPO_EPOCHS

    # --------------------------------------------------------
    # Select action
    # --------------------------------------------------------
    def select_action(self, obs):
        """
        Given a single observation (shape: (3,)),
        returns (action, log_prob, value_estimate).
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mu, std, v = self.net(obs_t)

        dist = torch.distributions.Normal(mu, std)
        act = dist.sample()

        # Clamp to a safe range
        act_clamped = torch.clamp(act, -1.0, 1.0)

        logp = dist.log_prob(act).sum(dim=-1)

        return act_clamped.item(), logp.item(), v.item()

    # --------------------------------------------------------
    # PPO Update
    # --------------------------------------------------------
    def update(self, rollouts: dict):
        """
        rollouts dict should contain:
            "obs"      : list/array of observations
            "actions"  : list/array of actions
            "logp"     : list/array of old log-probs
            "rewards"  : list/array of rewards
            "values"   : list/array of value estimates
            "dones"    : list/array of done flags (0/1)
        """
        obs = torch.tensor(np.array(rollouts["obs"]), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(rollouts["actions"]), dtype=torch.float32, device=DEVICE)
        logp_old = torch.tensor(np.array(rollouts["logp"]), dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(np.array(rollouts["rewards"]), dtype=torch.float32, device=DEVICE)
        values = torch.tensor(np.array(rollouts["values"]), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(np.array(rollouts["dones"]), dtype=torch.float32, device=DEVICE)

        T = len(rewards)
        adv = torch.zeros(T, device=DEVICE)

        # ----------------- GAE -----------------
        last_gae = 0.0
        last_val = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_val * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            adv[t] = last_gae
            last_val = values[t]

        returns = adv + values

        # ----------------- PPO Update -----------------
        for _ in range(self.epochs):
            mu, std, v_pred = self.net(obs)
            dist = torch.distributions.Normal(mu, std)
            logp = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(logp - logp_old)
            surr = ratio * adv

            policy_loss = -surr.mean()
            value_loss = 0.5 * (returns - v_pred).pow(2).mean()

            loss = policy_loss + value_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
