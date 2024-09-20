from typing import Sequence, cast
import numpy as np
import torch
from torch.cuda import CUDAGraph
import torch.nn as nn
import torch.nn.functional as F

from blocks import Actor, ObsNormalizer, QNetwork

class SAC(nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        use_obs_norm: bool,
    ) -> None:
        super().__init__()

        # constants
        self.gamma = gamma
        self.tau = tau

        # init models
        self.actor = Actor(obs_size, act_size).to(device)
        self.qf1 = QNetwork(obs_size, act_size).to(device)
        self.qf2 = QNetwork(obs_size, act_size).to(device)
        self.qf1_target = QNetwork(obs_size, act_size).to(device)
        self.qf2_target = QNetwork(obs_size, act_size).to(device)

        # copy targets
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # entropy tuning
        self.target_entropy = -act_size
        self.log_alpha = nn.Parameter(
            torch.tensor(0.0, requires_grad=True, device=device)
        )

        # optimizers
        self.critic_optim = torch.optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=critic_lr,
            capturable=True,
        )
        self.actor_optim = torch.optim.Adam(
            list(self.actor.parameters()),
            lr=actor_lr,
            capturable=True,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
            amsgrad=True,
            capturable=True,
        )

        # obs normalizer
        if use_obs_norm:
            self.obs_normalizer = ObsNormalizer(
                obs_size=obs_size
            ).to(device)
        else:
            self.obs_normalizer = None

        # compile
        self.compile()
        self.cuda_graph: None | torch.cuda.CUDAGraph = None
        self.infos_ref: None | dict[str, torch.Tensor] = None
        self.batch_ref: None | Sequence[torch.Tensor] = None

    def update(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: torch.Tensor
    ) -> dict[str, float]:
        # stack everything so it's easier to handle
        batch = (obs, act, rew, term, next_obs)

        if self.cuda_graph is None:
            # sample a batch, make the batch ref, copy to batch ref
            self.batch_ref = [torch.zeros_like(x) for x in batch]
            [t.copy_(s) for s, t in zip(batch, self.batch_ref)]

            # warmup step
            self.infos_ref = self.forward(*self.batch_ref)

            # construct the graph
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                self.infos_ref = self.forward(*self.batch_ref)
            torch.cuda.synchronize()

        # cast some things so pyright doesn't scream
        self.cuda_graph = cast(CUDAGraph, self.cuda_graph)
        self.infos_ref = cast(dict[str, torch.Tensor], self.infos_ref)
        self.batch_ref = cast(Sequence[torch.Tensor], self.batch_ref)

        # start the training!
        [t.copy_(s) for s, t in zip(batch, self.batch_ref)]
        self.cuda_graph.replay()

        # return the infos as a dict of floats
        return {k: float(v.cpu().numpy()) for k, v in self.infos_ref.items()}

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        term: torch.Tensor,
        next_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # normalize_obs
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
            next_obs = self.obs_normalizer.normalize(next_obs)

        # compute target value
        with torch.no_grad():
            next_act, next_log_pi = self.actor.get_action(next_obs)
            qf1_next_target = self.qf1_target(next_obs, next_act)
            qf2_next_target = self.qf2_target(next_obs, next_act)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.log_alpha.exp() * next_log_pi
            )
            next_q_value = rew.flatten() + (
                1 - term.flatten()
            ) * self.gamma * (min_qf_next_target).view(-1)

        # compute critic loss
        qf1_a_values = self.qf1(obs, act).view(-1)
        qf2_a_values = self.qf2(obs, act).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the critic
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # compute actor loss
        pi, log_pi = self.actor.get_action(obs)
        qf1_pi = self.qf1(obs, pi)
        qf2_pi = self.qf2(obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.log_alpha.exp() * log_pi) - min_qf_pi).mean()

        # optimize actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # compute alpha loss
        with torch.no_grad():
            _, log_pi = self.actor.get_action(obs)
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

        # optimize alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # update the target networks
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # log some things
        log = dict()
        log["algorithm/critic_loss"] = qf_loss.mean().detach()
        log["algorithm/actor_loss"] = actor_loss.mean().detach()
        log["algorithm/alpha_loss"] = alpha_loss.mean().detach()
        log["algorithm/target_q"] = next_q_value.mean().detach()
        log["algorithm/log_pi"] = log_pi.mean().detach()

        return log
