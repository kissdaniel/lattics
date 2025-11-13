import numpy as np
import pandas as pd
import pickle
import csv

from .time_series import TimeSeries
from lattics.core import Simulation


def get_agent_numbers_from_file(filename: str, filter=None) -> TimeSeries:
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


def get_agent_numbers_from_history(simulation: Simulation, filter=None) -> TimeSeries:
    times = []
    values = []
    history = simulation._history
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
