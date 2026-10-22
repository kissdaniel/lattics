def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    TIME_UNITS = {
        'ms': 1,
        'sec': 1000,
        'min': 60 * 1000,
        'hour': 60 * 60 * 1000,
        'day': 24 * 60 * 60 * 1000,
        'week': 7 * 24 * 60 * 60 * 1000,
    }
    if from_unit not in TIME_UNITS:
        raise ValueError(
            f"Invalid time unit: '{from_unit}'. Supported units are: 'ms', 'sec', 'min', 'hour', 'day', 'week'."
        )
    if to_unit not in TIME_UNITS:
        raise ValueError(
            f"Invalid time unit: '{to_unit}'. Supported units are: 'ms', 'sec', 'min', 'hour', 'day', 'week'."
        )
    return value * TIME_UNITS[from_unit] / TIME_UNITS[to_unit]


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    LENGTH_UNITS = {'um': 1, 'mm': 1000, 'cm': 10 * 1000}
    if from_unit not in LENGTH_UNITS:
        raise ValueError(f"Invalid length unit: '{from_unit}'. Supported units are: 'um', 'mm', 'cm'.")
    if to_unit not in LENGTH_UNITS:
        raise ValueError(f"Invalid length unit: '{to_unit}'. Supported units are: 'um', 'mm', 'cm'.")
    return value * LENGTH_UNITS[from_unit] / LENGTH_UNITS[to_unit]


def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
    VOLUME_UNITS = {
        'um3': 1,
        'mm3': 10**9,
        'cm3': 10**12,
        'nL': 10**6,
        'uL': 10**9,
        'mL': 10**12,
    }
    if from_unit not in VOLUME_UNITS:
        raise ValueError(
            f"Invalid volume unit: '{from_unit}'. Supported units are: 'um3', 'mm3', 'cm3', 'nL', 'uL', 'mL'."
        )
    if to_unit not in VOLUME_UNITS:
        raise ValueError(
            f"Invalid volume unit: '{to_unit}'. Supported units are: 'um3', 'mm3', 'cm3', 'nL', 'uL', 'mL'."
        )
    return value * VOLUME_UNITS[from_unit] / VOLUME_UNITS[to_unit]


def convert_velocity(value: float, from_unit: str, to_unit: str) -> float:
    VELOCITY_UNITS = {
        'um/ms': 1,
        'um/sec': 1000,
        'um/min': 60 * 1000,
        'um/hour': 60 * 60 * 1000,
        'mm/ms': 1000,
        'mm/sec': 1000 * 1000,
        'mm/min': 1000 * 60 * 1000,
        'mm/hour': 1000 * 60 * 60 * 1000,
        'cm/ms': 10 * 1000,
        'cm/sec': 10 * 1000 * 1000,
        'cm/min': 10 * 1000 * 60 * 1000,
        'cm/hour': 10 * 1000 * 60 * 60 * 1000,
    }
    if from_unit not in VELOCITY_UNITS:
        raise ValueError(
            f"Invalid velocity unit: '{from_unit}'. Supported units are: 'um/ms', 'um/sec', 'um/min', 'um/hour', "
            "'mm/ms', 'mm/sec', 'mm/min', 'mm/hour', 'cm/ms', 'cm/sec', 'cm/min', 'cm/hour'."
        )
    if to_unit not in VELOCITY_UNITS:
        raise ValueError(
            f"Invalid velocity unit: '{to_unit}'. Supported units are: 'um/ms', 'um/sec', 'um/min', 'um/hour', "
            "'mm/ms', 'mm/sec', 'mm/min', 'mm/hour', 'cm/ms', 'cm/sec', 'cm/min', 'cm/hour'."
        )
    return value * VELOCITY_UNITS[from_unit] / VELOCITY_UNITS[to_unit]


_TIME_UNITS = {
    'ms': 1,
    'sec': 1000,
    'min': 60 * 1000,
    'hour': 60 * 60 * 1000,
    'day': 24 * 60 * 60 * 1000,
    'week': 7 * 24 * 60 * 60 * 1000,
}
_LENGTH_UNITS = {'um': 1, 'mm': 1000, 'cm': 10 * 1000}
_RATE_UNITS = {f'1/{t}': 1 / factor for t, factor in _TIME_UNITS.items()}
_DIFFUSIVITY_UNITS = {
    f'{length}2/{t}': length_factor**2 / time_factor
    for length, length_factor in _LENGTH_UNITS.items()
    for t, time_factor in _TIME_UNITS.items()
}


def convert_rate(value: float, from_unit: str, to_unit: str) -> float:
    """Converts a rate (an amount per time) between units of the form ``'1/<time unit>'``, for example
    ``convert_rate(0.1, '1/hour', '1/ms')``.
    """
    for unit in (from_unit, to_unit):
        if unit not in _RATE_UNITS:
            raise ValueError(f"Invalid rate unit: '{unit}'. Supported units are: {', '.join(_RATE_UNITS)}.")
    return value * _RATE_UNITS[from_unit] / _RATE_UNITS[to_unit]


def convert_diffusivity(value: float, from_unit: str, to_unit: str) -> float:
    """Converts a diffusion coefficient between units of the form ``'<length unit>2/<time unit>'``, for example
    ``convert_diffusivity(2.0, 'um2/sec', 'um2/ms')``.
    """
    for unit in (from_unit, to_unit):
        if unit not in _DIFFUSIVITY_UNITS:
            raise ValueError(
                f"Invalid diffusion coefficient unit: '{unit}'. Use '<length>2/<time>' with the length units "
                f"{', '.join(_LENGTH_UNITS)} and the time units {', '.join(_TIME_UNITS)}, for example 'um2/sec'."
            )
    return value * _DIFFUSIVITY_UNITS[from_unit] / _DIFFUSIVITY_UNITS[to_unit]
