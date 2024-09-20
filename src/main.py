import tempfile
import time

import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memorial.replay_buffers import FlatReplayBuffer
from wingman import Wingman
from pathlib import Path
import shimmy
gym.register_envs(shimmy)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env

    return thunk


class ObsNormalizer(nn.Module):
    def __init__(self, obs_size: int) -> None:
        super().__init__()
        self._count = nn.Parameter(
            torch.tensor(0, dtype=torch.int64),
            requires_grad=False,
        )
        self._mean = nn.Parameter(
            torch.zeros(
                size=(obs_size,),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self._var = nn.Parameter(
            torch.ones(
                size=(obs_size,),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @property
    def mean(self) -> np.ndarray:
        return self._mean.detach().cpu().numpy()

    @property
    def var(self) -> np.ndarray:
        return self._var.detach().cpu().numpy()

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._mean) / (self._var + 1e-3).sqrt()

    def forward(self, obs: torch.Tensor) -> None:
        self.update(obs)

    def update(self, obs: torch.Tensor) -> None:
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        batch_count = obs.shape[0]
        new_count = self._count + batch_count

        # compute new mean
        batch_mean = obs.mean(dim=0)
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * batch_count / new_count

        # compute new variance
        batch_var = obs.var(dim=0, unbiased=False)
        new_var = (
            self._var * (self._count / new_count)
            + batch_var * (batch_count / new_count)
            + delta.pow(2) * (self._count / new_count) * (batch_count / new_count)
        )

        # update parameters
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)
        self._count.copy_(new_count)


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

    """SETUPS"""
    # env setup
    train_envs = gym.vector.SyncVectorEnv([make_env(wm.cfg.env_id)])
    eval_envs = gym.vector.SyncVectorEnv([make_env(wm.cfg.env_id)])
    assert isinstance(train_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # alg setup
    actor = Actor(train_envs).to(wm.device)
    qf1 = SoftQNetwork(train_envs).to(wm.device)
    qf2 = SoftQNetwork(train_envs).to(wm.device)
    qf1_target = SoftQNetwork(train_envs).to(wm.device)
    qf2_target = SoftQNetwork(train_envs).to(wm.device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    target_entropy = -torch.prod(torch.Tensor(train_envs.single_action_space.shape).to(wm.device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=wm.device)
    alpha = log_alpha.exp().item()

    critic_optim = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=wm.cfg.q_lr)
    actor_optim = optim.Adam(list(actor.parameters()), lr=wm.cfg.policy_lr)
    alpha_optim = optim.Adam([log_alpha], lr=wm.cfg.q_lr)

    # replay buffer setup
    rb = FlatReplayBuffer(wm.cfg.buffer_size)
    global_start_time = time.time()

    # obs normalizer
    if wm.cfg.obs_norm:
        obs_normalizer = ObsNormalizer(obs_size=int(train_envs.single_observation_space.shape))
    else:
        obs_normalizer = None

    """START TRAINING"""
    obs, _ = train_envs.reset()
    has_reset = False
    evaluation_score = 0.0
    for global_step in tqdm(range(wm.cfg.total_timesteps)):
        loop_timer = time.time()

        # update obs normalizer
        if obs_normalizer is not None:
            obs_normalizer.update(obs)

        # sample an action
        if global_step < wm.cfg.learning_starts:
            act = np.array([train_envs.single_action_space.sample() for _ in range(train_envs.num_envs)])
        else:
            # to tensor and conditionally normalize
            t_obs = torch.Tensor(obs).to(wm.device)
            if obs_normalizer is not None:
                t_obs = obs_normalizer.normalize(t_obs)

            act, _, _ = actor.get_action(t_obs)
            act = act.detach().cpu().numpy()

        # take an environment step
        next_obs, rew, term, trunc, _ = train_envs.step(act)

        # add to buffer if we're not in a fresh state
        if not has_reset:
            rb.push([obs, act, rew, term, next_obs], bulk=True)

        # rollover things
        obs = next_obs
        has_reset = term[0] or trunc[0]

        # log some things
        wm.log["timers/loop_time"] = time.time() - loop_timer

        # perform evaluation if needed
        if global_step % wm.cfg.evaluation_freq == 0:
            loop_timer = time.time()
            eval_obs, _ = eval_envs.reset()
            eval_done = False
            evaluation_score = 0.0
            while not eval_done:
                # to tensor and conditionally normalize
                t_obs = torch.Tensor(obs).to(wm.device)
                if obs_normalizer is not None:
                    t_obs = obs_normalizer.normalize(t_obs)

                eval_act, _, _ = actor.get_action(t_obs)
                eval_act = eval_act.detach().cpu().numpy()

                # take an environment step
                eval_next_obs, eval_rew, eval_term, eval_trunc, _ = eval_envs.step(eval_act)
                evaluation_score += float(eval_rew[0])

                # rollover things
                eval_obs = eval_next_obs
                eval_done = eval_term[0] or eval_trunc[0]

            # log some things
            wm.log["timers/eval_time"] = time.time() - loop_timer
            wm.log["evaluation/cumulative_reward"] = evaluation_score

        # take a gradient step if allowed
        if global_step > wm.cfg.learning_starts:
            loop_timer = time.time()

            # sample and send to device
            s_obs, s_act, s_rew, s_term, s_next_obs = rb.sample(wm.cfg.batch_size)
            s_obs = torch.tensor(s_obs, device=wm.device)
            s_act = torch.tensor(s_act, device=wm.device)
            s_rew = torch.tensor(s_rew, device=wm.device)
            s_term = torch.tensor(s_term, device=wm.device)
            s_next_obs = torch.tensor(s_next_obs, device=wm.device)

            # normalize_obs
            if obs_normalizer is not None:
                s_obs = obs_normalizer.normalize(s_obs)

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

            # log some things
            wm.log["timers/update_time"] = time.time() - loop_timer
            wm.log["algorithm/critic_loss"] = float(qf_loss.mean().detach().cpu())
            wm.log["algorithm/actor_loss"] = float(actor_loss.mean().detach().cpu())
            wm.log["algorithm/alpha_loss"] = float(alpha_loss.mean().detach().cpu())
            wm.log["algorithm/target_q"] = float(next_q_value.mean().detach().cpu())
            wm.log["algorithm/log_pi"] = float(log_pi.mean().detach().cpu())

        wm.checkpoint(loss=-evaluation_score, step=global_step)

    train_envs.close()

    if obs_normalizer is None:
        exit()

    if not wm.cfg.wandb.enable:
        exit()

    with tempfile.NamedTemporaryFile(suffix=".png") as mean_f, tempfile.NamedTemporaryFile(suffix=".png") as var_f:
        # plot and save mean
        plt.bar(
            range(obs_normalizer.mean.shape[0]),
            obs_normalizer.mean,
        )
        plt.title(f"Mean - {wm.cfg.env_id}")
        plt.savefig(mean_f.name)
        plt.close()

        # plot and save var
        plt.bar(
            range(obs_normalizer.var.shape[0]),
            obs_normalizer.var,
        )
        plt.title(f"Var - {wm.cfg.env_id}")
        plt.savefig(var_f.name)
        plt.close()

        # log to wandb
        wandb.log(
            {
                "obs_mean": wandb.Image(mean_f.name),
                "obs_var": wandb.Image(var_f.name),
            }
        )
