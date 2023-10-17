"""A Class to identify peaks in time dependent data."""
import numpy as np


class PeakDetector:
    """
    Class to determine real time peak detection signals.

    This class is built using the following post:
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362
    """

    def __init__(self, array, lag, threshold, influence):
        """
        Initialize the class.

        Args:
        array: Numpy 1D array containing the data
        lag: The lag of the moving window
        threshold: The threshold. When abs(zscore) > threshold, signal
        influence: The influence (between 0 and 1)

        Returns:
        None
        """
        self.y = list(array)
        self.length = len(self.y)
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.signals = [0] * len(self.y)
        self.filteredY = np.array(self.y).tolist()
        self.avgFilter = [0] * len(self.y)
        self.stdFilter = [0] * len(self.y)
        self.avgFilter[self.lag - 1] = np.mean(self.y[0 : self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0 : self.lag]).tolist()

    def update_params(self, influence=None, threshold=None):
        """
        Update the influence and threshold parameters.

        Args:
        influence: The influence (between 0 and 1)
        threshold: The threshold. When abs(zscore) > threshold, signal

        Returns:
        None
        """
        if influence is not None:
            self.influence = influence
        if threshold is not None:
            self.threshold = threshold

    def thresholding_algo(self, new_value):
        """
        Algorithm to determine whether the next value is a signal.

        Args:
        new_value: Next value to process

        Returns:
        Tuple of 3 numpy arrays: (signals, filteredY, num_std_deviation)
        """
        # add the new value to the window
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)

        # if the window is not full, return 0
        if i < self.lag:
            return (0, 0, 0)
        # once the window is full, initialize the signals, filteredY, avgFilter, and stdFilter
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag] = np.mean(self.y[0 : self.lag]).tolist()
            self.stdFilter[self.lag] = np.std(self.y[0 : self.lag]).tolist()
            self.iqr = [0] * len(self.y)
            return (0, 0, 0)

        # prepare to process the next value
        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]
        self.iqr += [0]

        # if the difference between the new value and the average of the window
        # is greater than the threshold times the standard deviation of the
        # window, then it is a signal
        if abs((self.y[i] - self.avgFilter[i - 1])) > (
            self.threshold * self.stdFilter[i - 1]
        ):
            # if the new value is greater than the average, then it is a positive signal
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            # currently not using negative signals, but could be used in the future
            else:
                self.signals[i] = 0
            # update the filtered values using the influence
            self.filteredY[i] = (
                self.influence * self.y[i]
                + (1 - self.influence) * self.filteredY[i - 1]
            )

        # if the new value is not a signal, then the filtered value is just the new value
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
        self.avgFilter[i] = np.mean(self.filteredY[i - self.lag : i])
        self.stdFilter[i] = np.std(self.filteredY[i - self.lag : i])

        # return the signals, filteredY, and the number of standard deviations
        return (
            self.signals[i],
            self.avgFilter[i],
            abs(self.y[i] - self.avgFilter[i - 1]) / self.stdFilter[i - 1]
            if self.stdFilter[i - 1] != 0
            else 0,
        )
