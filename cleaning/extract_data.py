"""
Extract data from a relevant time frame.
"""

import os
import logging
import pandas as pd


class Subsetter:
    """Read the messy data, extract a fitting time range and write to disk."""

    def __init__(self):
        self.data = None
        self.stations = None
        self.dates = None
        self.idx = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logfile = os.path.join("logs", "extract_data.log")
        file_handler = logging.FileHandler(logfile)
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.logger.info("%s Start extraction %s", 10 * "=", 10 * "=")

    def read(self, path, **kwargs):
        """Wraps `pandas.read_csv`"""
        self.data = pd.read_csv(path, **kwargs)

    def get_index(self):
        """Extract stations in the raw data and create a `pd.MultiIndex`."""
        self.stations = self.data.index.get_level_values("station").unique()
        self.dates = pd.date_range("2008/04/14", "2017/12/10")
        self.idx = pd.MultiIndex.from_product([self.stations, self.dates])

    def reindex(self):
        """Reindex the raw data.

        Ensure there is a row for each station for each day.
        """
        self.data = self.data.reindex(self.idx)

    def clean(self):
        """Remove stations with missing values """
        na_counts = self.data.groupby(level=1).PRCP.count()
        check_stations = na_counts == 3528
        whitelist = check_stations.loc[check_stations].index.values
        ok_rows_idx = self.data.index.get_level_values(0).isin(whitelist)
        self.data = self.data.loc[ok_rows_idx]
        self.check_integrity()

    def check_integrity(self):
        """Check if number of station and number of days is as expected"""
        n_stations = self.data.index.get_level_values(0).nunique()
        n_days = self.data.index.get_level_values(1).nunique()
        try:
            assert n_days == 3528, "Wrong number of days"
            assert n_stations == 2642, "Wrong number of stations"
            self.logger.info("Everything as expected")

        except AssertionError as exc:
            self.logger.exception(exc)

    def write(self, path, **kwargs):
        """Write data to disk."""
        self.data.to_csv(path, **kwargs)


def main():
    """Read data, extract relevant data and write to disk."""
    datapath = os.path.join("data", "ghcnd", "ghcnd_08_18_all.csv")
    outpath = os.path.join("data", "ghcnd", "ghcnd.csv")

    subsetter = Subsetter()
    subsetter.read(datapath, parse_dates=[0], index_col=[1, 0], header=0)
    subsetter.get_index()
    subsetter.reindex()
    subsetter.clean()
    subsetter.write(outpath, index_label=["station", "date"])


if __name__ == "__main__":
    main()
