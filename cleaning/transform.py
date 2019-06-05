"""
Routines for cleaning the GHDC-weather dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FH = logging.FileHandler(os.path.join("logs", "transform.log"))
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FORMATTER = logging.Formatter(FORMAT)
FH.setFormatter(FORMATTER)
LOGGER.addHandler(FH)
LOGGER.info("Start new run".center(49, "="))

DATAPATH = os.path.join("data", "ghcnd", "ghcnd_all")
OUTPATH = os.path.join("data", "ghcnd", "cleaned")
MANIFEST = os.path.join("data", "ghcnd", "data_overview.csv")
FILELIST = glob(os.path.join(DATAPATH, "*.dly"))

ID_FIELDS = (0, 21)
VAL_FIELDS = [(l, l+5) for l in range(21, 269, 8)]
M_FIELDS = [(l, l+1) for l in range(26, 269, 8)]
Q_FIELDS = [(l, l+1) for l in range(27, 269, 8)]
S_FIELDS = [(l, l+1) for l in range(28, 269, 8)]

COLSPECS = sorted([ID_FIELDS, *VAL_FIELDS, *M_FIELDS, *Q_FIELDS, *S_FIELDS])


def clean(filename):
    """Restructure a given file and perform preliminary cleaning."""
    dirty = pd.read_fwf(filename, colspecs=COLSPECS, index_col=0,
                        header=None, na_values=[-9999])
    dirty = dirty.iloc[:, ::4]
    try:
        assert dirty.shape[1] == 31
    except AssertionError:
        raise AssertionError("Number of columns after pivoting is not 31.")

    dirty.columns = range(1, 32)

    # Fix the index
    id_vars = dirty.index.str.extract(
        r"^([_A-Za-z0-9\-]{11})(\d{4})(\d{2})(.*)$",
        expand=True).set_index(dirty.index)
    id_vars.columns = ["station", "year", "month", "measure"]

    dirty = pd.concat([id_vars, dirty], axis=1)

    # Fix the columns
    long_dirty = pd.melt(dirty, var_name="day",
                         id_vars=["station", "year", "month", "measure"])

    try:
        assert long_dirty.station.nunique() == 1
    except AssertionError as e:
        raise AssertionError("Station name is not unique")
        LOGGER.error(e)
    station_name = long_dirty.station.unique()[0]

    year_str = long_dirty.year.astype(str)
    month_str = long_dirty.month.astype(str)
    day_str = long_dirty.day.astype(str).str.pad(2, fillchar="0")

    long_dirty["id"] = year_str + "-" + month_str + "-" + day_str

    # Reformat into a more sensible structure
    out = long_dirty.pivot(index="id", columns="measure", values="value")
    columns = out.columns.to_list()

    # Remove non-existant calendar dates, find start and stop by sorting values
    date_ranges = long_dirty.loc[:, ["year", "month", "day"]].sort_values(
        ["year", "month", "day"]).iloc[[0, -1]]
    date_range_flag = False
    while not date_range_flag:
        try:
            start, end = pd.to_datetime(date_ranges)
            valid_dates = pd.date_range(start, end)
            date_range_flag = True
            assert len(valid_dates) > 0
        except ValueError:
            date_ranges.iloc[-1, 2] -= 1

    valid_rows = out.index.isin(valid_dates.astype(str))
    out = out.iloc[valid_rows, :]
    out = out[~out.isnull().all(axis=1)]
    out.index.name = "date"
    out["station"] = station_name

    return out[["station"] + columns]


def diagnose(data):
    """Diagnose, summarize and report characteristics of a dataframe."""
    n, m = data.shape
    LOGGER.info("Number of Rows: %s", n)
    LOGGER.info("Number of Columns: %s", m)

    cols = [c for c in data.columns if c != "station"]
    LOGGER.info("Set of variables:" + " %s"*len(cols), *cols)

    type_title_flag = False
    for c in cols:
        try:
            data[c].astype(float)
        except ValueError:
            if not type_title_flag:
                type_title_flag = True
                LOGGER.warning("Some columns are non-numerical:")

            v_counts = data[c].value_counts()
            LOGGER.warning("\n" + c.center(39, "-") + "\n%s", v_counts)

    if not type_title_flag:
        LOGGER.info("All columns are numerical")

    na_counts = data[cols].isna().sum()
    nrow = data.shape[0]
    start_date, end_date = data.index.sort_values().values[[0, -1]]
    station = data.station.unique()[0]

    diag = na_counts.reset_index()
    diag.columns = ["column", "na_count"]
    diag["station"] = station
    diag["start_date"] = start_date
    diag["end_date"] = end_date
    diag["nrow"] = nrow
    outcols = ["station", "start_date", "end_date",
               "column", "nrow", "na_count"]

    return diag.loc[:, outcols]


def main(filename):
    csv_name = filename.replace(".dly", ".csv").replace(DATAPATH+"/", "")
    LOGGER.info(csv_name.center(49, "-"))

    try:
        out = clean(filename)
        diag = diagnose(out)
        diag["file"] = csv_name
        out.to_csv(os.path.join(OUTPATH, csv_name))
        if not os.path.exists(MANIFEST):
            diag.to_csv(MANIFEST, index=False)
        elif os.path.exists(MANIFEST):
            diag.to_csv(MANIFEST, header=False, mode="a", index=False)

    except Exception as e:
        LOGGER.error(e)


if __name__ == "__main__":
    for f in tqdm(FILELIST):
        main(f)
