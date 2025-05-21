#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 8 
#SBATCH --mem=16g 
#SBATCH -J "rl2048" 
#SBATCH -p short 
#SBATCH -t 18:00:00 
#SBATCH --output=logs/R-%x_%j.out
#SBATCH --error=logs/R-%x_%j.err

export PYTHONUNBUFFERED=1
uv run main.py --seed 42 --tuples "Matsuzaki" -lr 0.1 -tc 0.1 --n-iterations 5000 --n-episodes 100 --save-interval 100 --load -o Standard_lr_1e-1 --no-tqdm 
