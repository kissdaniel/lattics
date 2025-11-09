import numpy as np
import pandas as pd
import csv

from .util import convert_time


class TimeSeries:
    def __init__(self, timepoints, values, time_unit='ms'):
        self.timepoints = timepoints
        self.values = values
        self.time_unit = time_unit

    def get_timepoints(self, time_unit: str = None) -> list[float]:
        if not time_unit:
            return self.timepoints
        return [convert_time(t, self.time_unit, time_unit) for t in self.timepoints]

    def get_values(self) -> list[float]:
        return self.values

    def sse(self, other: 'TimeSeries', interpolation: str = 'no') -> float:
        if interpolation == 'no':
            return self._sse_nointerpolation(other)

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

            for i, row in enumerate(reader, start=2 if skip_header else 1):
                time = int(row[0].strip())
                value = float(row[1].strip())

                timepoints.append(time)
                values.append(value)

        return TimeSeries(timepoints, values, time_unit)

    def _sse_nointerpolation(self, other: 'TimeSeries') -> float:
        i = 0
        j = 0
        total_squared_diff = 0.0
        while i < len(self.timepoints) and j < len(other.timepoints):
            self_timepoint = convert_time(self.timepoints[i], self.time_unit, 'ms')
            other_timepoint = convert_time(other.timepoints[j], other.time_unit, 'ms')
            if self_timepoint < other_timepoint:
                i += 1
            elif other_timepoint < self_timepoint:
                j += 1
            else:
                diff = self.values[i] - other.values[j]
                total_squared_diff += diff ** 2
                i += 1
                j += 1
        return total_squared_diff
