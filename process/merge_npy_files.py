import numpy as np
from glob import glob
import sys, os


def list_files(run, dipole):
    """prepare sorted file names"""

    files = glob(f"mesh_{dipole}_run{run}_*.npy")
    files_dict = {}
    for fname in files:
        part = int(fname.split('_')[-1].split('.')[0])
        files_dict[part] = fname

    return sorted(files_dict.items())


def merge(flist):
    """load into numpy arrays and merge"""

    # db
    print(f"run {run}, {len(flist)} batches")
    ####
    merged = np.load(flist[0][1])
    os.remove(flist[0][1])

    for f in flist[1:]:
        try:
            merged_ = np.load(f[1])
        except:
            print(f"failed at {f[1]}")
            break
        os.remove(f[1])
        merged = np.vstack((merged, merged_))

    return merged


def main():
    global run
    run = int(sys.argv[1])

    for dipole in ['top', 'bottom']:
        list_of_files = list_files(run, dipole)
        merged = merge(list_of_files)
        np.save(f"mesh_{dipole}_run{run}.npy", merged)


if __name__ == '__main__':
    main()