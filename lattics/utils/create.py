from lattics.core import Agent
from lattics.core import Simulation


def create_agent(**params):
    agent = Agent()
    for par_name, par_value in params.items():
        if not agent.has_attribute(par_name):
            agent.set_attribute(par_name, par_value)
    return agent


def create_population(size: int):
    agent = Agent()
    return [agent]


def create_simulation(id: str = None):
    return Simulation(id=id)
