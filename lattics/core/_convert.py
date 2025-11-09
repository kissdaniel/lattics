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
