"""
Implement Classic Timeseries Decomposition class with predict method.

This method is outdated. Apperently so outdated that the `statsmodels` guys
cannot be bothered to implement a predict method. Which is fair. In general,
you should use more sophisticated methods. But I just need a very simple
baseline for more complicated models.

This class implements the classic/ancient way of decomposing a timeseries into
a trend, a seasonal component and the error. Also it is able to extrapolate the
estimated trend into the future.

Note however, that the class expects daily data, and estimates the seasonal
component as the average deviation from the trend, grouped by month. This may
or may not suit your needs.
"""

import numpy as np
import pandas as pd
from tqdm import trange
from matplotlib import pyplot as plt

# pylint: disable=no-member


class ClassicTSDecomposition:
    """Classic Decomposition of Timeseries into Trend, Season and Residuals."""

    def __init__(self):
        """Initialize."""
        self.intercept = None
        self.slope = None
        self.n_observed = None
        self.season = None
        self.data = None
        self.prediction = None

    def fit(self, data):
        """
        Fit model to data

        Parameters
        ----------

        * data : `pandas.Series`
             `pandas.Series` of data. Index must be a daily `pd.DatetimeIndex`.
        """

        self.n_observed = len(data)
        self.data = pd.DataFrame(data)
        self.data.columns = ["x"]
        self.data["t"] = range(self.n_observed)
        self.data["month"] = self.data.index.month
        self.intercept, self.slope, self.data["trend"] = self.__get_trend()
        self.season = self.__get_seasonal()
        self.data = pd.merge(self.data.reset_index(),
                             self.season).set_index("index")
        self.data["fitted"] = self.data.trend + self.data.season
        self.data["residuals"] = self.data.x - self.data.fitted

    def __get_trend(self):
        intercept, slope = self.__compute_coefficients()
        trend = self.__compute_trend(intercept, slope, self.data.t)
        return intercept, slope, trend

    def __compute_coefficients(self):
        n = self.n_observed
        sum_x, sum_t = self.data["x"].sum(), self.data["t"].sum()
        sum_t_sq = (self.data["t"]**2).sum()
        sum_x_times_t = (self.data["x"] * self.data["t"]).sum()
        denom = n * sum_t_sq - sum_t**2

        intercept = (sum_x * sum_t_sq - sum_t * sum_x_times_t) / denom
        slope = (n * sum_x_times_t - sum_x * sum_t) / denom
        return intercept, slope

    @staticmethod
    def __compute_trend(intercept, slope, time):
        trend = intercept + slope * time
        return trend

    def __get_seasonal(self):
        diff_to_trend = self.data.x - self.data.trend
        seasonal = diff_to_trend.groupby(diff_to_trend.index.month).mean()
        out = pd.DataFrame({"month": seasonal.index.values,
                            "season": seasonal.values})
        return out

    def predict(self, steps):
        """
        Predict `steps` steps ahead.

        Parameters
        ----------

        * `steps` : `int`
            Forecast horizon. I.e. number of days to be predicted.

        Returns
        -------

        * `self.prediction.x` : `pandas.Series`
            Series of Predictions. Has `len(self.prediction.x) = steps`.
        """

        time = np.arange(self.n_observed + 1, self.n_observed + steps + 1)
        last_idx = self.data.index.max()
        idx = [last_idx + pd.Timedelta(days=i) for i in range(1, steps+1)]
        pred_trend = self.__compute_trend(self.intercept, self.slope, time)
        self.prediction = pd.DataFrame(pred_trend,
                                       index=idx,
                                       columns=["trend"])
        self.prediction["month"] = self.prediction.index.month
        self.prediction = pd.merge(self.prediction.reset_index(),
                                   self.season, on="month").set_index("index")
        self.prediction["x"] = self.prediction.trend + self.prediction.season
        return self.prediction.x

    def plot(self, **kwargs):
        """
        Draw diagnostic plots.

        Parameters
        ----------

        * `kwargs` :
             Keyword arguments to `matplotlib.pyplot.subplots`.

        Returns

        * `fig`, `axes` :
            Figures and axes of plots.
        """

        fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, **kwargs)
        cols = [["x", "fitted"], "trend", "season", "residuals"]
        labels = ["overview", "trend", "season", "residuals"]
        for ax, col, lab in zip(axes, cols, labels):
            self.data.loc[:, col].plot(ax=ax)
            ax.set_title(lab)

        fig.tight_layout()
        return fig, axes


class ClassicTSDecompositionTester:
    """
    Test Suite for ClassicTSDecomposition class.

    The tests are mostly bullshit.
    """

    def __init__(self):
        self.n_observedests = []
        self.intercept_tests = []
        self.slope_tests = []
        self.seasonal_tests = []

    @staticmethod
    def get_test_data(intercept, slope, season_range=50, years=3):
        """Obtain test data with known trend and season."""
        num = 365 * years
        date_range = pd.date_range("2001-01-01", periods=num)
        season = np.random.uniform(-season_range, season_range, size=12)
        season = season - season.mean()
        values = {"t": range(num),
                  "trend": [intercept + slope*t for t in range(num)],
                  "season": [season[t-1] for t in date_range.month],
                  "residual": np.random.normal(size=num)}
        data = pd.DataFrame(values, index=date_range)
        data["x"] = data.trend + data.season + data.residual
        series = data.loc[:, "x"]
        seasonal = pd.Series(season, index=date_range.month.unique())
        return series, seasonal

    def run_tests(self, num):
        """
        Run `num` tests.

        Keep in mind however, that the tests are bullshit.
        """
        for i in trange(num):
            print(20*"-" + f" Test {i+1:02d} " + 20*"-")
            params = {"intercept": np.random.uniform(-10, 10),
                      "slope": np.random.random(),
                      "years": np.random.randint(1, 100)}
            series, seasonal = self.get_test_data(**params)
            model = ClassicTSDecomposition()
            model.fit(series)

            print("Intercept | Model: {0:10.5f} | True: {1:10.5f}".format(
                model.intercept, params["intercept"]))
            print("Slope     | Model: {0:10.5f} | True: {1:10.5f}".format(
                model.slope, params["slope"]))

            pass_intercept = np.isclose(params["slope"],
                                        model.slope, 0.3)
            pass_slope = np.isclose(params["slope"], model.slope, 0.1)
            zip_true_model_season = zip(seasonal, model.season.season)
            s_test = [np.isclose(t, m, 2) for t, m in zip_true_model_season]
            pass_seasonal = all(s_test)
            pass_test = all([pass_intercept, pass_slope, pass_seasonal])
            self.n_observedests.append(pass_test)
            self.intercept_tests.append(pass_intercept)
            self.slope_tests.append(pass_slope)
            self.seasonal_tests.append(pass_seasonal)

    def summarize(self):
        """Summarize test results"""
        results = [self.n_observedests,
                   self.intercept_tests,
                   self.slope_tests,
                   self.seasonal_tests]
        test_types = [" ", " intercept ", " slope ", " seasonal "]

        for res, test_type in zip(results, test_types):
            n = len(res)
            n_passed = sum(res)
            print(f"PASSED {n_passed} out of {n}{test_type}tests.")


if __name__ == "__main__":
    TEST_SUITE = ClassicTSDecompositionTester()
    TEST_SUITE.run_tests(100)
    TEST_SUITE.summarize()
