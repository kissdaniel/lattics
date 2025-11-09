import abc
from lattics.core._agent import Agent
from lattics.core._updateinfo import UpdateInfo


class BaseModel(abc.ABC):
    """Base class for models.

    This class is intended for internal use only and should not be accessed directly by library users.
    """

    def __init__(self,
                 update_interval: tuple[float, str] = None
                 ) -> None:
        self.update_info = UpdateInfo(update_interval=update_interval)

    @abc.abstractmethod
    def initialize_attributes(self, agent: Agent, **params) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update_attributes(self, agent: Agent) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def reset_attributes(self, agent: Agent) -> None:
        raise NotImplementedError
