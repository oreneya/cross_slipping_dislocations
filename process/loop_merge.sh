#!/bin/bash

# go to folder
cd $1

for ((i=1; i<=1000; i++)); do
  file="mesh_top_run${i}_*.npy"
  if compgen -G $file > /dev/null; then
    echo "processing run ${i}"
    python ../../merge_npy_files.py ${i}
  fi
done
