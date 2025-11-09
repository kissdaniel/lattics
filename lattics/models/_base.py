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

    def initialize_attributes(self, agent: Agent) -> None:
        self._initialize_required_attributes(agent)
        self._initialize_optional_attributes(agent)
        self._initialize_attributes_default_values(agent)

    @abc.abstractmethod
    def update_attributes(self, agent: Agent) -> None:
        raise NotImplementedError

    def reset_attributes(self, agent: Agent) -> None:
        pass

    def _initialize_required_attributes(self, agent: Agent) -> None:
        pass

    def _initialize_optional_attributes(self, agent: Agent) -> None:
        pass

    def _initialize_attributes_default_values(self, agent: Agent) -> None:
        pass
