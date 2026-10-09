#!/usr/bin/env bash
#SBATCH --job-name=baseonly16_eval
#SBATCH --partition=L40
#SBATCH --account=p_p25505
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:l40:8
#SBATCH --time=48:00:00
#SBATCH --output=/share/home/u25505/apps/wangbo/LatentVLA/results/libero-plus/spatial_baseonly16_noaux_25k30k_l40_20261010/slurm-%j.out
#SBATCH --error=/share/home/u25505/apps/wangbo/LatentVLA/results/libero-plus/spatial_baseonly16_noaux_25k30k_l40_20261010/slurm-%j.err

set -Eeuo pipefail
mkdir -p /share/home/u25505/apps/wangbo/LatentVLA/results/libero-plus/spatial_baseonly16_noaux_25k30k_l40_20261010
exec /share/home/u25505/apps/wangbo/LatentVLA/LaRA-VLA/scripts/launch_baseonly16_noaux_eval_l40_20261010.sh
