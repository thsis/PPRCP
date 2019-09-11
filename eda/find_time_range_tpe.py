"""
Find the most densely populated time range in the data/

I.e. there are many stations, who collect their data on one year, but not on
the next. Preferably I would like to use the date range that contains most of
the stations' observations.

Idea:
    use `hyperopt` with an objective function that calculates the number of
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
import datetime
import logging
from pprint import pprint

import numpy as np
import pandas as pd

from hyperopt import hp, tpe, fmin, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from matplotlib import pyplot as plt


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
FILE_HANDLER = logging.FileHandler(os.path.join("logs", "find_time_range.log"))
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
FORMATTER = logging.Formatter(FORMAT)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)
LOGGER.info("%s Start new run %s", 10 * "=", 10 * "=")
LOGGER.info("Start, Stop, Score")

DATADIR = os.path.join("data", "ghcnd", "ghcnd_08_18_all.csv")
OUTDIR = os.path.join("eda", "select.csv")
FIGPATH = os.path.join("figures", "find_time_range.png")

DATA = pd.read_csv(DATADIR, usecols=[0, 1, 2],
                   parse_dates=[0], names=["day", "station", "prcp"],
                   header=0, index_col=[1, 0])

STATIONS = DATA.index.get_level_values(0).unique()
START = DATA.index.get_level_values(1).min()
END = DATA.index.get_level_values(1).max()

START_RANGE = pd.date_range(START, END-pd.Timedelta(days=365 * 3))
END_RANGE = pd.date_range(START+pd.Timedelta(days=365 * 3), END)

DATES = list(itertools.product(START_RANGE, END_RANGE))


def objective(params):
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
    start_y = params["start_y"]
    start_m = params["start_m"]
    start_d = params["start_d"]
    stop_y = params["stop_y"]
    stop_m = params["stop_m"]
    stop_d = params["stop_d"]
    try:
        start_dt = datetime.date(year=int(start_y),
                                 month=int(start_m),
                                 day=int(start_d))
        stop_dt = datetime.date(year=int(stop_y),
                                month=int(stop_m),
                                day=int(stop_d))
        start, stop = pd.to_datetime([start_dt, stop_dt])
        date_range = pd.date_range(start, stop)
        days = len(date_range)
        idx = pd.MultiIndex.from_product([STATIONS, date_range])

        relevant = DATA[DATA.index.isin(idx)]
        relevant = relevant.reindex(idx)

        summary = relevant.groupby(level=[0]).count()
        stations = (summary.prcp >= days).sum()
        score = stations * days
        LOGGER.info("%s, %s, %s", start, stop, score)
        return -score
    except ValueError:
        return 0


def find_min():
    """Minimize Objective Function."""
    space = {
        "start_y": scope.int(hp.quniform("start_y", 2008, 2015, 1)),
        "start_m": scope.int(hp.quniform("start_m", 1, 12, 1)),
        "start_d": scope.int(hp.quniform("start_d", 1, 31, 1)),
        "stop_y": scope.int(hp.quniform("stop_y", 2010, 2019, 1)),
        "stop_m": scope.int(hp.quniform("stop_m", 1, 12, 1)),
        "stop_d": scope.int(hp.quniform("stop_d", 1, 31, 1))}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)
    return best, trials


def visualize(trials):
    """Visualize Search."""
    vals = np.abs(trials.losses())
    fig, ax = plt.subplots()
    ax.plot(vals)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Value")
    ax.set_title("Objective Function")
    return fig, ax


if __name__ == "__main__":
    BEST, TRIALS = find_min()
    pprint(BEST)
    LOGGER.info(10 * "-" + " Best run " + "-" * 10)
    LOGSTR_1 = "From: {start_d:.0f}-{start_m:.0f}-{start_y:.0f}".format(**BEST)
    LOGSTR_2 = "To: {stop_d:.0f}-{stop_m:.0f}-{stop_y:.0f}".format(**BEST)
    LOGGER.info(LOGSTR_1)
    LOGGER.info(LOGSTR_2)
    FIG, AX = visualize(TRIALS)
    FIG.savefig(FIGPATH)
