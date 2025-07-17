#!/bin/bash
# WANDB_DISABLED=true python train.py --task=T1 --num_envs=1 --headless=1 --sim_device=cpu --rl_device=cpu

# python train.py --task=T1 --num_envs=2 

python train.py --task=T1 --headless=1 --num_envs=64
