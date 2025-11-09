from ._convert import convert_time


class UpdateInfo:
    def __init__(self,
                 update_interval: tuple[float, str]
                 ) -> None:
        if update_interval:
            self._update_interval = convert_time(update_interval[0], update_interval[1], 'ms')
        else:
            self._update_interval = 0
        self._elapsed_time = 0

    @property
    def update_interval(self) -> int:
        return self._update_interval

    @property
    def elapsed_time(self) -> int:
        return self._elapsed_time

    def update_needed(self) -> bool:
        return self._update_interval <= self._elapsed_time

    def increase_time(self, msecs) -> None:
        self._elapsed_time += msecs

    def reset_time(self) -> None:
        self._elapsed_time = 0
