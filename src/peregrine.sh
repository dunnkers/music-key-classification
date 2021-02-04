#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=16000
#SBATCH --ntasks=10
#SBATCH --partition=regular
#SBATCH --job-name=key-recognition
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=1,3,5,8,12,15
# → do make sure /logs directory exists!

module load Python/3.8.2-GCCcore-9.3.0
pip3 install -r requirements.txt --user
python3 src/peregrine.py --n_components ${SLURM_ARRAY_TASK_ID}

