#!/bin/bash
#
#SBATCH --job-name=addition_ponders
#SBATCH --mem=12G
#SBATCH --gres=gpu:1,gmem:11G
#SBATCH --array=1-20
srun python addition_test.py --ponder $SLURM_ARRAY_TASK_ID
