#!/bin/bash -l
#SBATCH -o ./output/run%j.output
#SBATCH -e ./output/run%j.error
#SBATCH --export=ALL
#SBATCH --ntasks 4
#SBATCH --ntasks-per-node 2

LMP=/efs-mount/LAMMPS/lammps-patch_15Jun2023/src/lmp_mpi
S=${2}0
T=$3

if test -f ./output/restart_run$1; then
    mpirun $LMP -sf opt -var TEMP $T -var STRESS $S -var rand $1 -in in.run_cont
else
    mpirun $LMP -sf opt -var TEMP $T -var STRESS $S -var rand $1 -in in.run
fi
