#!/bin/bash

#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH -w gnode025
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vennam404@gmail.com
#SBATCH --job-name=per_fetch
#SBATCH --time=4-00:00:00

eval "$(conda shell.bash hook)"       # Activate the conda environment
conda activate nlp

# set envs
export TRANSFORMERS_CACHE="/scratch/vamshi.b/.cache"
export HF_DATASETS_CACHE="/scratch/vamshi.b/.cache"

# run necessary commands
# python scripts/multi_rot_generation.py data/quality100/q100_gpt_3.5.csv data/quality100/q100_gpt_3.5_multi_rot.csv
python scripts/edge_generation.py data/quality100/q100_gpt_3.5_pairs.csv data/quality100/context_edges/q100_gpt_3.5_cross_ar.csv
