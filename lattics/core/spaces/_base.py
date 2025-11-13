from abc import ABC, abstractmethod

from lattics.core import Agent
from lattics.core import Simulation
from lattics.core import UpdateInfo


class BaseSpace(ABC):
    """Base class for different simulation domains.
    """
    def __init__(self,
                 simulation: Simulation,
                 dt_agent: tuple[float, str] = None,
                 dt_substrate: tuple[float, str] = None,
                 ) -> None:
        """Constructor method.

        Parameters
        ----------
        simulation : Simulation
            The Simulation instance containing the domain
        """
        self._simulation = simulation
        self._update_infos = dict()
        self._agent_update_info = UpdateInfo(dt_agent)
        self._substrate_update_info = UpdateInfo(dt_substrate)

    @abstractmethod
    def add_agent(self, agent: Agent) -> None:
        """Adds the specified agent to the structure representing the internal
        storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be added
        """
        pass

    @abstractmethod
    def remove_agent(self, agent: Agent) -> None:
        """Removes the specified agent from the internal storage of the domain.

        Parameters
        ----------
        agent : Agent
            The agent to be removed
        """
        pass

    @abstractmethod
    def add_substrate(self,
                      name: str,
                      diffusion_coefficient: float = 0.0,
                      decay_coefficient: float = 0.0,
                      decay_kinetics: str = 'first-order'
                      ) -> None:
        pass

    @abstractmethod
    def update(self, dt: int) -> None:
        """Updates the domain according to the specified time duration.

        Parameters
        ----------
        dt : int
            The time elapsed since the last update call, in milliseconds
        """
        pass
