#!/bin/bash

stress=0
temperature=450

for i in {1..2}; do
  sbatch -p simulations -J "run:$run" go_run.sh $run $stress $temperature
  sbatch -p postprocess go_process.sh $run
done