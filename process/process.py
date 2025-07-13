import os
from glob import glob
import sys
import time
import subprocess


def get_job_arg(job_id):
    """
    Find number of run with which the job was assigned
    """
    res = subprocess.run(['scontrol', 'show', 'jobid', '-dd', job_id], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if res.startswith('slurm_load_jobs error'):
        return 'Job does not exist'
    res = res.split('\n')
    namePair = res[0].split()[1]
    name = namePair.split('=')[1]
    run = name.split(':')[1]
    return run


def job_exists(run):
    """
    Determine if a job is still running
    """
    res = subprocess.run(['squeue'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    lines = res.split()[1:]
    runspec = "run:" + run
    for line in lines:
        if line.find(runspec) > -1:
            return 1
    return 0


def main():

    run = sys.argv[1]
    path = './output'
    
    # determine initial batch number
    num_mesh_files = len(glob(f"{path}/results/mesh_top_run{run}_*"))
    batch = num_mesh_files + 1

    # check for triger every 10 minutes
    while job_exists(run):
    
        # count number of dump files
        dump_files = glob(f"{path}/dump.full_run{run}.*0.gz")
        nof = len(dump_files)
        print("Dump files:", nof)
    
        # apply dxa on the first 100 files = 1ps = 0.56gb
        if nof > 1: #00:
            # sort list by timestep
            sorted_files = []
            for filepath in dump_files:
                zfilled = filepath.split('.')[-2].zfill(6)
                sorted_files.append([zfilled, filepath])
            sorted_files.sort()
            # apply dxa
            for i, flname in enumerate(sorted_files):
                os.system(f"bash go_dxa.sh {flname[1]} {run}")
    
        else:
            time.sleep(600)
            continue
    
        # apply extract_segments.py
        os.system(f"bash go_extract.sh {path} {run} {batch}")
        batch += 1


if __name__ == '__main__':
    main()
