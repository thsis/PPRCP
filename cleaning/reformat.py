"""
Reformat the .dly files to .csv files.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def reformat_dly(filename):
    """Insert commata at specific positions of each line of a file."""
    element_idx = 21
    val_idx = np.arange(26, 269, 8)
    mflag_idx = np.arange(27, 269, 8)
    qflag_idx = np.arange(28, 269, 8)
    sflag_idx = np.arange(29, 269, 8)

    comma_idx = [element_idx, *val_idx, *mflag_idx, *qflag_idx, *sflag_idx]
    positions = sorted(comma_idx, reverse=True)

    sed_command = "sed -i '" + "s/./&,/{};" * len(positions) + "' {}"
    os.system(sed_command.format(*positions, filename))


if __name__ == "__main__":
    datapath = os.path.join("data", "ghcnd")
    fp = open(os.path.join(datapath, "country_codes.txt"))
    groups = fp.read().splitlines()
    files = [os.path.join(datapath, "ghcnd_all", f+"*.dly") for f in groups]
    fp.close()

    Parallel(n_jobs=-1)(delayed(reformat_dly)(g) for g in tqdm(files))
