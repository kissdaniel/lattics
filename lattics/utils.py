import numpy as np
import pandas as pd
import pickle


class UnitConverter:
    TIME_UNITS = {
        'ms': 1,
        'sec': 1000,
        'min': 60 * 1000,
        'hour': 60 * 60 * 1000,
        'day': 24 * 60 * 60 * 1000,
        'week': 7 * 24 * 60 * 60 * 1000
    }

    @staticmethod
    def time_to_ms(expression: tuple[float, str]) -> int:
        value = expression[0]
        unit = expression[1]
        if unit in UnitConverter.TIME_UNITS:
            return value * UnitConverter.TIME_UNITS[unit]
        else:
            raise ValueError(f"Invalid time unit: '{unit}'. Supported units are: 'ms', 'sec', 'min', 'hour', 'day', 'week'.")

    @staticmethod
    def ms_to_days(value: int) -> float:
        return value / (24 * 60 * 60 * 1000)


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


def get_agent_numbers(filename: str, filter=None):
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
    return (times, values)


def get_substrate_concentration(filename: str, name: str, means_only: bool = False, **params):
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
    return (times, values)


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
