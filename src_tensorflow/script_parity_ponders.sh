#!/bin/bash
#
#SBATCH --job-name=parity_ponders
#SBATCH --mem=6G
#SBATCH --gres=gpu:1,gmem:11G
#SBATCH --array=1-20
srun python parity_test.py --ponder $SLURM_ARRAY_TASK_ID
