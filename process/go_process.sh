#!/bin/bash -l
#SBATCH -J process
#SBATCH -o ./output/process%j.output
#SBATCH -e ./output/process%j.error
#SBATCH --export=ALL

# process a selected simulation with the provided id
python3 process.py $1
