# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memorial.replay_buffers import FlatReplayBuffer
from wingman import Wingman
from pathlib import Path


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    wm = Wingman(Path(__file__).parent / "config.yaml")

    # seeding
    random.seed(wm.cfg.seed)
    np.random.seed(wm.cfg.seed)
    torch.manual_seed(wm.cfg.seed)
    torch.backends.cudnn.deterministic = wm.cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """SETUPS"""
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(wm.cfg.env_id, wm.cfg.seed)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # alg setup
    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()

    critic_optim = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=wm.cfg.q_lr)
    actor_optim = optim.Adam(list(actor.parameters()), lr=wm.cfg.policy_lr)
    alpha_optim = optim.Adam([log_alpha], lr=wm.cfg.q_lr)

    # replay buffer setup
    rb = FlatReplayBuffer(wm.cfg.buffer_size)
    start_time = time.time()

    """START TRAINING"""
    obs, _ = envs.reset(seed=wm.cfg.seed)
    has_reset = False
    for global_step in range(wm.cfg.total_timesteps):
        # sample an action
        if global_step < wm.cfg.learning_starts:
            act = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            act, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            act = act.detach().cpu().numpy()

        # take an environment step
        next_obs, rew, term, truncations, infos = envs.step(act)

        # add to buffer if we're not in a fresh state
        if not has_reset:
            rb.push([obs, act, rew, term, next_obs])

        # rollover things
        obs = next_obs
        has_reset = term[0] or truncations[0]

        # take a gradient step
        if global_step > wm.cfg.learning_starts:
            # sample and send to device
            s_obs, s_act, s_rew, s_term, s_next_obs = rb.sample(wm.cfg.batch_size)
            s_obs = torch.tensor(s_obs, device=device)
            s_act = torch.tensor(s_act, device=device)
            s_rew = torch.tensor(s_rew, device=device)
            s_term = torch.tensor(s_term, device=device)
            s_next_obs = torch.tensor(s_next_obs, device=device)

            # compute target value
            with torch.no_grad():
                s_next_act, next_log_pi, _ = actor.get_action(s_next_obs)
                qf1_next_target = qf1_target(s_next_obs, s_next_act)
                qf2_next_target = qf2_target(s_next_obs, s_next_act)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                next_q_value = s_rew.flatten() + (1 - s_term.flatten()) * wm.cfg.gamma * (min_qf_next_target).view(-1)

            # compute critic loss
            qf1_a_values = qf1(s_obs, s_act).view(-1)
            qf2_a_values = qf2(s_obs, s_act).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the critic
            critic_optim.zero_grad()
            qf_loss.backward()
            critic_optim.step()

            # compute actor loss
            pi, log_pi, _ = actor.get_action(s_obs)
            qf1_pi = qf1(s_obs, pi)
            qf2_pi = qf2(s_obs, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

            # optimize actor
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # compute alpha loss
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(s_obs)
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            # optimize alpha
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            alpha = log_alpha.exp().item()

            # update the target networks
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(wm.cfg.tau * param.data + (1 - wm.cfg.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(wm.cfg.tau * param.data + (1 - wm.cfg.tau) * target_param.data)

    envs.close()
