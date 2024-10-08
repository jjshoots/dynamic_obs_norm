#!/bin/bash

source .venv/bin/activate

declare -a pids=()

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/acrobot-swingup-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/acrobot-swingup_sparse-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/ball_in_cup-catch-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-balance-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-balance_sparse-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-swingup-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-swingup_sparse-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-two_poles-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cartpole-three_poles-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/cheetah-run-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/dog-stand-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/dog-walk-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/dog-trot-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/dog-run-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/dog-fetch-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/finger-spin-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/finger-turn_easy-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/finger-turn_hard-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/fish-upright-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/fish-swim-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/hopper-stand-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/hopper-hop-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid-stand-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid-walk-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid-run-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid-run_pure_state-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid_CMU-stand-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid_CMU-walk-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/humanoid_CMU-run-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/lqr-lqr_2_1-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/lqr-lqr_6_2-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/manipulator-bring_ball-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/manipulator-bring_peg-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/manipulator-insert_ball-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/manipulator-insert_peg-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/pendulum-swingup-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/point_mass-easy-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/point_mass-hard-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/quadruped-walk-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/quadruped-run-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/quadruped-escape-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/quadruped-fetch-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/reacher-easy-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/reacher-hard-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/stacker-stack_2-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/stacker-stack_4-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/swimmer-swimmer6-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/swimmer-swimmer15-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done

python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/walker-stand-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/walker-walk-v0" &
pids+=($!)
python3 /home/taijunjet/Sandboxes/dynamic_obs_norm/src/main.py --wandb.enable --env_id="dm_control/walker-run-v0" &
pids+=($!)
for pid in ${pids[*]}; do
  wait $pid
done
