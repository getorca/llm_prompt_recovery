#!/bin/bash
#SBATCH --gpus=2
#SBATCH --partition=a100-long
#SBATCH --time=1-23:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output jlab-%J.log

eval "$(conda shell.bash hook)"
conda activate lm

CUDA_VISIBLE_DEVICES=0,1 python3 infer.py