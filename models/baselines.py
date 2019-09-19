"""
Define Baselines

1. global average
2. station average
3. station lagged value - 1 day
4. station lagged value - 1 year
5. classical time series decomposition
"""

import os

import pandas as pd
import numpy as np

from models.classic_tsa import ClassicTSDecomposition
from models.utilities import get_logger


def get_mock_data():
    """Create test data set."""
    datapath = os.path.join("data", "ghcnd", "ghcnd_08_18_all.csv")
    data = pd.read_csv(datapath, nrows=10_000, index_col=[1, 0],
                       parse_dates=[0], usecols=[0, 1, 2])
    stations = data.index.get_level_values(0).unique()
    dates = data.index.get_level_values(1)
    daterange = pd.date_range(dates.min(), dates.max())
    idx = pd.MultiIndex.from_product([stations, daterange],
                                     names=["station", "level_1"])
    data = data.reindex(idx).fillna(0)
    return data.PRCP


def split_train_validation(data):
    """Split into train and validation set"""
    dates = data.index.get_level_values(1)
    train_end = pd.to_datetime("2017-01-01")
    train = data.loc[dates < train_end]
    val = data.loc[dates >= train_end]
    return train, val


class Baseline:
    """Base class for Baselines."""
    name = ""

    def __init__(self):
        """Provide common attributes."""
        self.pred = None
        self.score = None
        self.train = None
        self.val = None

    def get_validation_score(self, val):
        """
        Compute mean squared error for a given validation set.

        Parameters
        ----------

        * `val`: `pd.Series`
            Validation set.

        """

        self.val = val
        self.score = np.nanmean((self.pred - self.val)**2)
        return self.score

    def fit(self, data):
        """Fit baseline to data"""
        raise NotImplementedError

    def predict(self, val_idx):
        """Predict values for a given Station/Date `pd.MultiIndex`"""
        raise NotImplementedError

    def test(self):
        """Run tests."""
        data = get_mock_data()
        train, val = split_train_validation(data)
        self.fit(train)
        self.predict(val.index)
        score = self.get_validation_score(val)
        return score

    def run(self, train, val, logger=None):
        """Train, Predict and compute a Validation score."""
        self.fit(train)
        self.predict(val.index)
        score = self.get_validation_score(val)
        if logger:
            logger.info(f"{self.name:<25s} {score:10.3f}")
        return score


class BaselineGlobalAvg(Baseline):
    """
    Always predict the global average.
    """
    name = "Global Average"

    def __init__(self):
        super().__init__()
        self.global_avg = None

    def fit(self, data):
        """Fit Baseline to data.

        Parameters
        ----------
        * `data`: `pd.Series`
            Train data
        """

        self.train = data
        self.global_avg = data.mean()

    def predict(self, val_idx):
        """Predict specific days.

        Parameters
        ----------
        * `val_idx`: `pd.MultiIndex`
            Station/Day combinations to be predicted.
        """
        self.pred = pd.Series(index=val_idx)
        self.pred.fillna(self.global_avg, inplace=True)
        return self.pred


class BaselineStationAvg(Baseline):
    """
    Always predict the station's average.
    """

    name = "Station Average"

    def __init__(self):
        super().__init__()
        self.station_avg = None

    def fit(self, data):
        """Fit Baseline to data.

        Parameters
        ----------
        * `data`: `pd.Series`
            Train data
        """
        self.train = data
        self.station_avg = self.train.groupby(level=0).mean()

    def predict(self, val_idx):
        """Predict specific days.

        Parameters
        ----------
        * `val_idx`: `pd.MultiIndex`
            Station/Day combinations to be predicted.
        """
        self.pred = self.station_avg.reindex(val_idx, level="station")
        return self.pred


class BaselineLagged(Baseline):
    """Predict lagged values"""

    def __init__(self, lagged_days=1):
        super().__init__()
        self.lagged_days = lagged_days
        self.name = f"Lag by {lagged_days} day(s)"

    def fit(self, data):
        """Fit Baseline to data.
        Parameters
        ----------
        * `data`: `pd.Series`
            Train data
        """

        self.train = data

    def predict(self, val_idx):
        """Predict specific days.

        Parameters
        ----------
        * `val_idx`: `pd.MultiIndex`
            Station/Day combinations to be predicted.
        """
        self.pred = self.train.shift(self.lagged_days)
        self.pred = self.pred[self.pred.index.isin(val_idx)]
        return self.pred

    def test(self):
        """Run tests."""
        data = get_mock_data()
        _, val = split_train_validation(data)
        self.fit(data)
        self.predict(val.index)
        score = self.get_validation_score(val)
        return score


class BaselineTSDecomp(Baseline):
    """Predict based on Classic Time Series Decomposition"""

    name = "Classic TS Decomposition"

    def __init__(self):
        super().__init__()
        self.pred = []
        self.models = {}

    def fit(self, data):
        """Fit Baseline to data.
        Parameters
        ----------
        * `data`: `pd.Series`
            Train data
        """
        self.train = data.reset_index()
        for station in self.train.station.unique():
            subset = self.__get_subset(station, self.train)

            model = ClassicTSDecomposition()
            model.fit(subset)
            self.models[station] = model

    @staticmethod
    def __get_subset(station, data):
        subset = data.loc[data.station == station]
        subset.set_index("level_1", inplace=True)
        subset.index.rename("index", inplace=True)
        return subset.loc[:, "PRCP"]

    def predict(self, val_idx):
        """Predict specific days.

        Parameters
        ----------
        * `val_idx`: `pd.MultiIndex`
            Station/Day combinations to be predicted.
        """
        steps, stations = self.__preprocess_predictions(val_idx)

        for station in stations:
            predictions = self.__get_predictions(station, steps)
            self.pred.append(predictions)

        self.__postprocess_predictions()
        return self.pred

    def __preprocess_predictions(self, val_idx):
        steps = self.__get_num_steps(val_idx)
        stations = self.__get_stations(val_idx)
        return steps, stations

    def __postprocess_predictions(self):
        self.pred = pd.concat(self.pred)
        self.pred.columns = ["level_1", "x", "station"]
        self.pred.set_index(["station", "level_1"], inplace=True)
        self.pred = self.pred.x

    def __get_predictions(self, station, steps):
        model = self.models[station]
        predictions = model.predict(steps)
        predictions = predictions.reset_index()
        predictions["station"] = station
        return predictions

    @staticmethod
    def __get_num_steps(val_idx):
        steps = val_idx.get_level_values(1).nunique()
        return steps

    @staticmethod
    def __get_stations(val_idx):
        stations = val_idx.get_level_values(0).unique()
        return stations


class BaselineSuite:
    """Suite to run a flexible number Baseline objects."""

    def __init__(self):
        self.logger = get_logger(os.path.join("logs", "baseline.log"))
        self.logger.info("%s Compute Baselines %s", 10*"=", 10*"=")
        self.run_params = [(None, None, None)]

    def load(self, baselines, train_sets, val_sets):
        """Load instance with Baselines."""
        self.run_params = zip(baselines, train_sets, val_sets)

    def run(self):
        """Run a number of Baseline tests against a validation test."""
        for baseline, train, val in self.run_params:
            baseline.run(train=train, val=val, logger=self.logger)


def main():
    """Run Multiple Baselines and Log the Results."""
    data = get_mock_data()
    train, val = split_train_validation(data)
    params = {"baselines": [BaselineGlobalAvg(),
                            BaselineStationAvg(),
                            BaselineLagged(1),
                            BaselineLagged(365),
                            BaselineTSDecomp()],
              "train_sets": [train, train, data, data, train],
              "val_sets": [val] * 5}

    baseline_suite = BaselineSuite()
    baseline_suite.load(**params)
    baseline_suite.run()


if __name__ == "__main__":
    main()
