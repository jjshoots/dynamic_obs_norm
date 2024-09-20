import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class QNetwork(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_size + act_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_size)
        self.fc_logstd = nn.Linear(256, act_size)
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x)

        # Enforcing Action Bound
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
