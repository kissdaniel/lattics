from typing import Iterator

from ._agent import Agent


class Population:
    """The collection of agents participating in a simulation. Every agent is
    stored here exactly once; the spaces only keep spatial indices.

    The population assigns the identifiers of its agents: an agent receives
    the next free identifier when it is added. Agents keep the order in which
    they were added.
    """

    def __init__(self) -> None:
        self._agents = dict()
        self._next_id = 0

    def add(self, agent: Agent) -> None:
        """Adds the agent and assigns its identifier.

        Parameters
        ----------
        agent : Agent
            The agent to be added

        Raises
        ------
        ValueError
            If the agent already belongs to a population
        """
        if agent.id is not None:
            raise ValueError(f'The agent already belongs to a population (id={agent.id}).')
        agent._id = self._next_id
        self._next_id += 1
        self._agents[agent.id] = agent

    def remove(self, agent: Agent) -> None:
        """Removes the agent.

        Parameters
        ----------
        agent : Agent
            The agent to be removed

        Raises
        ------
        KeyError
            If the agent is not part of the population
        """
        del self._agents[agent.id]

    def __contains__(self, agent: Agent) -> bool:
        return self._agents.get(agent.id) is agent

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[Agent]:
        # Iterates over a snapshot, so agents can be added or removed meanwhile
        return iter(list(self._agents.values()))
