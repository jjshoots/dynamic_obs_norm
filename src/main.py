import tempfile
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import shimmy
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from memorial.replay_buffers import FlatReplayBuffer
from tqdm import tqdm
from wingman import Wingman

from blocks import Actor, ObsNormalizer, QNetwork
from sac import SAC

gym.register_envs(shimmy)


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env

    return thunk


if __name__ == "__main__":
    wm = Wingman(Path(__file__).parent / "config.yaml")

    """SETUPS"""
    # env setup
    train_envs = gym.vector.SyncVectorEnv([make_env(wm.cfg.env_id)])
    eval_envs = gym.vector.SyncVectorEnv([make_env(wm.cfg.env_id)])
    assert isinstance(
        train_envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # alg setup
    alg = SAC(
        train_envs.single_observation_space.shape[0],
        train_envs.single_action_space.shape[0],
        device=wm.device,
        critic_lr=wm.cfg.critic_lr,
        actor_lr=wm.cfg.actor_lr,
        alpha_lr=wm.cfg.critic_lr,
        gamma=wm.cfg.gamma,
        tau=wm.cfg.tau,
        use_obs_norm=wm.cfg.obs_norm,
    )
    # replay buffer setup
    rb = FlatReplayBuffer(wm.cfg.buffer_size)
    global_start_time = time.time()

    """START TRAINING"""
    obs, _ = train_envs.reset()
    has_reset = False
    evaluation_score = 0.0
    for global_step in tqdm(range(wm.cfg.total_timesteps)):
        # update obs normalizer
        if alg.obs_normalizer is not None:
            alg.obs_normalizer.update(torch.tensor(obs, device=wm.device))

        # sample an action
        if global_step < wm.cfg.learning_starts:
            act = np.array(
                [
                    train_envs.single_action_space.sample()
                    for _ in range(train_envs.num_envs)
                ]
            )
        else:
            # to tensor and conditionally normalize
            t_obs = torch.tensor(obs, device=wm.device, dtype=torch.float32)
            if alg.obs_normalizer is not None:
                t_obs = alg.obs_normalizer.normalize(t_obs)

            act, _ = alg.actor.get_action(t_obs)
            act = act.detach().cpu().numpy()

        # take an environment step
        next_obs, rew, term, trunc, _ = train_envs.step(act)

        # add to buffer if we're not in a fresh state
        if not has_reset:
            rb.push([obs, act, rew, term, next_obs], bulk=True)

        # rollover things
        obs = next_obs
        has_reset = term[0] or trunc[0]

        # perform evaluation if needed
        if global_step % wm.cfg.evaluation_freq == 0:
            eval_obs, _ = eval_envs.reset()
            eval_done = False
            evaluation_score = 0.0
            while not eval_done:
                # to tensor and conditionally normalize
                t_obs = torch.tensor(obs, device=wm.device, dtype=torch.float32)
                if alg.obs_normalizer is not None:
                    t_obs = alg.obs_normalizer.normalize(t_obs)

                eval_act, _ = alg.actor.get_action(t_obs)
                eval_act = eval_act.detach().cpu().numpy()

                # take an environment step
                eval_next_obs, eval_rew, eval_term, eval_trunc, _ = eval_envs.step(
                    eval_act
                )
                evaluation_score += float(eval_rew[0])

                # rollover things
                eval_obs = eval_next_obs
                eval_done = eval_term[0] or eval_trunc[0]

            # log some things
            wm.log["evaluation/cumulative_reward"] = evaluation_score

        # take a gradient step if allowed
        if global_step > wm.cfg.learning_starts:
            # sample and update and record infos
            update_infos = alg.update(
                *[torch.tensor(x, device=wm.device) for x in rb.sample(wm.cfg.batch_size)]
            )
            for k, v in update_infos.items():
                wm.log[f"algorithm/{k}"] = v

        # checkpoint things
        wm.checkpoint(loss=-evaluation_score, step=global_step)

    # gracefully close environments
    train_envs.close()
    eval_envs.close()

    # exit if no wandb or no normalizer
    if alg.obs_normalizer is None:
        exit()
    if not wm.cfg.wandb.enable:
        exit()

    with tempfile.NamedTemporaryFile(
        suffix=".png"
    ) as mean_f, tempfile.NamedTemporaryFile(suffix=".png") as var_f:
        # plot and save mean
        plt.bar(
            range(alg.obs_normalizer.mean.shape[0]),
            alg.obs_normalizer.mean,
        )
        plt.title(f"Mean - {wm.cfg.env_id}")
        plt.savefig(mean_f.name)
        plt.close()

        # plot and save var
        plt.bar(
            range(alg.obs_normalizer.var.shape[0]),
            alg.obs_normalizer.var,
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
