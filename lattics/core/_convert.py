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
            f"Invalid velocity unit: '{from_unit}'. Supported units are: 'um/ms', 'um/sec', 'um/min', 'um/hour', 'mm/ms', 'mm/sec', 'mm/min', 'mm/hour', 'cm/ms', 'cm/sec', 'cm/min', 'cm/hour'."
        )
    if to_unit not in VELOCITY_UNITS:
        raise ValueError(
            f"Invalid velocity unit: '{to_unit}'. Supported units are: 'um/ms', 'um/sec', 'um/min', 'um/hour', 'mm/ms', 'mm/sec', 'mm/min', 'mm/hour', 'cm/ms', 'cm/sec', 'cm/min', 'cm/hour'."
        )
    return value * VELOCITY_UNITS[from_unit] / VELOCITY_UNITS[to_unit]
