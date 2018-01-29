#!/bin/bash
#
#SBATCH --job-name=python
#SBATCH --mem=8G
#SBATCH --gres=gpu:1,gmem:11G
#SBATCH --array=1-12
srun python sort_test.py --ponder $SLURM_ARRAY_TASK_ID
