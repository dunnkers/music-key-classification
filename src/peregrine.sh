#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=8000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short
#SBATCH --job-name=key-recognition

module load Python/3.8.2-GCCcore-9.3.0
pip3 install -r requirements.txt --user
python3 src/key_recognition.py --verbose --give-mode hmm

