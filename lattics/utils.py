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
