#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --partition=regular
#SBATCH --job-name=key-recognition
#SBATCH --output=logs/slurm-%j.out
#SBATCH --array=1,3,5,8,12,15
# → do make sure /logs directory exists!

module load Python/3.8.2-GCCcore-9.3.0
pip3 install -r requirements.txt --user
python3 src/peregrine --n_components ${SLURM_ARRAY_TASK_ID}

