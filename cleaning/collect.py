"""
Collect data from 2008-2018
"""

import os
from glob import glob

import pandas as pd
from tqdm import tqdm
from joblib import delayed, Parallel

DATADIR = os.path.join("data", "ghcnd", "clean")
OUTDIR = os.path.join("data", "ghcnd", "ghcnd_08_18_all.csv")
FILELIST = glob(os.path.join(DATADIR, "*.csv"))

START = pd.to_datetime("2008-01-01")
END = pd.to_datetime("2018-12-31")
DATE_RANGE = pd.date_range(START, END)


def read_and_slice(filename):
    """Read dataframe and select the relevant timeframe."""
    out = pd.read_csv(filename, index_col=["date"], parse_dates=["date"])
    out = out.loc[out.index.isin(DATE_RANGE)]
    return out


def get_chunk(filelist, size=4):
    """
    Generator that iterates over a list in evenly sized chunks if possible.

    Arguments
    ---------
    `size` : `int`
        length of resulting chunks. Ideally this should correspond to the
        number of available cores on the machine.
    """
    for i in range(0, len(filelist), size):
        yield filelist[i:i+size]


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="Collect and store ghcnd data from 2008-2018")
    PARSER.add_argument("-n", "--n-jobs", type=int, default=4,
                        help="number of jobs")
    PARSER.add_argument("--dry-run", default=False, action="store_true",
                        help="if provided will not write to disk")
    ARGS = PARSER.parse_args(["--dry-run"])

    for chunk in tqdm(list(get_chunk(FILELIST, size=ARGS.n_jobs))):
        OUT = Parallel(n_jobs=ARGS.n_jobs)(delayed(read_and_slice)(f) for f in chunk)
        OUT = pd.concat(OUT, sort=False)
        if not ARGS.dry_run:
            OUT.to_csv(OUTDIR, mode="a")
