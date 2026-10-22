import csv

import numpy as np

from lattics._units import convert_time

# Tolerance for matching time points, in milliseconds
_TIME_TOLERANCE = 1e-6


class TimeSeries:
    """A series of values recorded at time points, for example by a recorder.

    Parameters
    ----------
    timepoints : list[float]
        The time points
    values : list
        The values
    time_unit : str, optional
        The unit of the time points; recorders use milliseconds
    """

    def __init__(self, timepoints, values, time_unit='ms'):
        self.timepoints = list(timepoints)
        self.values = list(values)
        self.time_unit = time_unit

    def __len__(self) -> int:
        return len(self.timepoints)

    def append(self, timepoint: float, value) -> None:
        """Appends a value recorded at the given time point."""
        self.timepoints.append(timepoint)
        self.values.append(value)

    def get_timepoints(self, time_unit: str = None) -> list[float]:
        if not time_unit:
            return self.timepoints
        return [convert_time(t, self.time_unit, time_unit) for t in self.timepoints]

    def get_values(self) -> list[float]:
        return self.values

    def sse(self, other: 'TimeSeries', interpolation: str = 'no') -> float:
        """The sum of squared differences at the time points the two series have in common."""
        if interpolation == 'no':
            return sum(diff**2 for diff in self._common_differences(other))
        raise ValueError(f'Unknown interpolation \'{interpolation}\'.')

    def rmse(self, other: 'TimeSeries', interpolation: str = 'no') -> float:
        """The root mean squared difference at the time points the two series have in common.

        Raises
        ------
        ValueError
            If the series have no time point in common
        """
        if interpolation != 'no':
            raise ValueError(f'Unknown interpolation \'{interpolation}\'.')
        differences = self._common_differences(other)
        if not differences:
            raise ValueError('The time series have no time point in common.')
        return float(np.sqrt(sum(diff**2 for diff in differences) / len(differences)))

    def to_csv(self, file_path: str, time_unit: str = None) -> None:
        """Writes the series to a CSV file with a header, readable with :meth:`from_csv`.

        Parameters
        ----------
        file_path : str
            The path of the file
        time_unit : str, optional
            The unit of the written time points; defaults to the unit of the series
        """
        unit = time_unit or self.time_unit
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([f'time [{unit}]', 'value'])
            for t, value in zip(self.get_timepoints(unit), self.values):
                writer.writerow([repr(float(t)), repr(value)])

    @staticmethod
    def from_csv(file_path: str, time_unit: str = 'ms', skip_header: bool = True) -> 'TimeSeries':
        timepoints = []
        values = []

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)

            if skip_header:
                try:
                    next(reader)
                except StopIteration:
                    return TimeSeries([], [], time_unit)

            for row in reader:
                timepoints.append(float(row[0].strip()))
                values.append(float(row[1].strip()))

        return TimeSeries(timepoints, values, time_unit)

    def _common_differences(self, other: 'TimeSeries') -> list[float]:
        i = 0
        j = 0
        differences = []
        while i < len(self.timepoints) and j < len(other.timepoints):
            self_timepoint = convert_time(self.timepoints[i], self.time_unit, 'ms')
            other_timepoint = convert_time(other.timepoints[j], other.time_unit, 'ms')
            if self_timepoint < other_timepoint - _TIME_TOLERANCE:
                i += 1
            elif other_timepoint < self_timepoint - _TIME_TOLERANCE:
                j += 1
            else:
                differences.append(self.values[i] - other.values[j])
                i += 1
                j += 1
        return differences
