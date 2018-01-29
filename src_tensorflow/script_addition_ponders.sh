#!/bin/bash
#
#SBATCH --job-name=python
#SBATCH --mem=12G
#SBATCH --gres=gpu:1,gmem:11G
#SBATCH --array=1-12
srun python addition_test.py --ponder $SLURM_ARRAY_TASK_ID
