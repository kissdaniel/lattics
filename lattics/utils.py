import numpy as np
import pandas as pd
import pickle
import csv


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


def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    TIME_UNITS = {
        'ms': 1,
        'sec': 1000,
        'min': 60 * 1000,
        'hour': 60 * 60 * 1000,
        'day': 24 * 60 * 60 * 1000,
        'week': 7 * 24 * 60 * 60 * 1000
    }
    if from_unit not in TIME_UNITS:
        raise ValueError(f"Invalid time unit: '{from_unit}'. Supported units are: 'ms', 'sec', 'min', 'hour', 'day', 'week'.")
    if to_unit not in TIME_UNITS:
        raise ValueError(f"Invalid time unit: '{to_unit}'. Supported units are: 'ms', 'sec', 'min', 'hour', 'day', 'week'.")
    return value * TIME_UNITS[from_unit] / TIME_UNITS[to_unit]


def export_agent_attributes(filename: str, attributes: list[str]):
    if not isinstance(attributes, list):
        attributes = [attributes]
    df_colnames = ['time', 'agent_id'] + attributes
    df_list = list()
    with open(filename, 'rb') as f:
        history = pickle.load(f)
        for entry in history:
            sim_state = pickle.loads(entry)
            agent_ids = [a.id for a in sim_state.agents]
            times = [sim_state.time for a in sim_state.agents]
            df_tmp = pd.DataFrame(columns=df_colnames)
            df_tmp['time'] = times
            df_tmp['agent_id'] = agent_ids
            for attr in attributes:
                attribs = [a.get_attribute(attr) for a in sim_state.agents]
                df_tmp[attr] = attribs
            df_list.append(df_tmp)
    return pd.concat(df_list, ignore_index=True)


def get_agent_numbers(filename: str, filter=None) -> TimeSeries:
    times = []
    values = []
    with open(filename, 'rb') as f:
        history = pickle.load(f)
        for entry in history:
            sim_state = pickle.loads(entry)
            times.append(sim_state.time)
            if filter:
                attribute = filter[0]
                condition = filter[1]
                values.append(sum([1 for a in sim_state.agents if a.get_attribute(attribute) == condition]))
            else:
                values.append(sum([1 for a in sim_state.agents]))
    return TimeSeries(times, values)


def get_substrate_concentration(filename: str, name: str, means_only: bool = False, **params) -> TimeSeries:
    times = []
    values = []
    with open(filename, 'rb') as f:
        history = pickle.load(f)
        for entry in history:
            sim_state = pickle.loads(entry)
            times.append(sim_state.time)
            if means_only:
                values.append(np.mean(sim_state._space._substrates[name]._concentration))
            else:
                if 'position' in params:
                    values.append(sim_state._space._substrates[name]._concentration[params['position']])
                else:
                    values.append(sim_state._space._substrates[name]._concentration)
    return TimeSeries(times, values)


def show_agents_and_substrates_2d(filename: str, times: tuple[int], substrate: str):
    import matplotlib.pyplot as plt
    counter = 0
    with open(filename, 'rb') as f:
        history = pickle.load(f)
        for entry in history:
            sim_state = pickle.loads(entry)
            if sim_state.time in times:
                fig, ax = plt.subplots()
                s = sim_state._space._substrates[substrate]._concentration
                s_ax = ax.imshow(s, vmin=0, vmax=0.035, cmap='inferno')
                for a in sim_state.agents:
                    pos = a.get_attribute('position')
                    color = 'blue' if a.get_attribute('state') == 'malignant' else 'red'
                    patch = plt.Circle(pos, 0.35, edgecolor='black', facecolor=color, linewidth=0.5)
                    ax.add_patch(patch)
                plt.axis('off')
                plt.colorbar(s_ax)
                plt.savefig(f'{counter}.png', dpi=100)
                plt.close()
                counter += 1
