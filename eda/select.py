"""
Find the most densely populated time range in the data/

I.e. there are many stations, who collect their data on one year, but not on
the next. Preferably I would like to use the date range that contains most of
the stations' observations.

Idea:
    brute-force an objective function that calculates the number of
    consecutive observations. Something like:
        U = (stop - start) * nr_stations

    where:
     * `stop`:
          is the first parameter consisting of YEAR-MONTH-DAY
     * `start`:
          is the second parameter consisting of YEAR-MONTH-DAY
     * `nr_stations`:
          is the number of stations that do not have any missing values between
          `start` and `stop`
"""

import os
import itertools

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


DATADIR = os.path.join("data", "ghcnd", "ghcnd_08_18_all.csv")
OUTDIR = os.path.join("eda", "select.csv")

DATA = pd.read_csv(DATADIR, usecols=[0, 1, 2],
                   parse_dates=[0], names=["day", "station", "prcp"],
                   header=0, index_col=[1, 0])

STATIONS = DATA.index.get_level_values(0).unique()
START = DATA.index.get_level_values(1).min()
END = DATA.index.get_level_values(1).max()

START_RANGE = pd.date_range(START, END-pd.Timedelta(days=365 * 3))
END_RANGE = pd.date_range(START+pd.Timedelta(days=365 * 3), END)

DATES = list(itertools.product(START_RANGE, END_RANGE))


def objective(start, stop):
    """
    Calculate number of usable cells for a given date range.

    Parameters
    ----------
    * `start`, `stop` : `pd.Timestamp`
        Beginning and end of interesting time period

    Returns
    -------
    * `out` : `tuple`
        Consists of (`score`, `stations`, `days`), where `stations` is the
        number of stations that do not contain any missing values in the
        period, `days` is the length of the time frame and score is calculated
        as `days` * `stations`.
    """
    date_range = pd.date_range(start, stop)
    days = len(date_range)
    idx = pd.MultiIndex.from_product([STATIONS, date_range])

    relevant = DATA[DATA.index.isin(idx)]
    relevant = relevant.reindex(idx)

    summary = relevant.groupby(level=[0]).count()
    stations = (summary.prcp >= days).sum()
    score = stations * days
    return score, stations, days


def main():
    """Brute-Force the objective function over the entire range."""
    res = Parallel(n_jobs=-1)(delayed(objective)(s, e) for s, e in tqdm(DATES))

    out = pd.DataFrame(res)
    out = pd.DataFrame(res, columns=["score", "num_stations", "num_days"])
    out.sort_values("score", ascending=False, inplace=True)
    out.to_csv(OUTDIR, index=None)


if __name__ == "__main__":
    main()
