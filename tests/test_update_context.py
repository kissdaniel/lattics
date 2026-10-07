import pytest


def _recording_model_class():
    from lattics.models import BaseModel

    class Recording(BaseModel):
        reads = ('space.volume',)

        def __init__(self, update_interval):
            super().__init__(update_interval)
            self.calls = []

        def update_agent(self, agent, ctx):
            self.calls.append((agent.id, ctx.time, ctx.dt))

    return Recording


def test_custom_model_needs_only_update_agent():
    from lattics import Agent, HomogeneousSpace, Simulation

    sim = Simulation()
    sim.set_space(HomogeneousSpace(dt_agent=(1, 'hour')))
    model = _recording_model_class()(update_interval=(1, 'hour'))
    sim.add_model(model)
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    sim.add_agent(agent)
    sim.run(time=(3, 'hour'), dt=(30, 'min'), verbosity=0)
    # the model runs every hour and receives the time elapsed since its previous update
    expected = [(0, 3600000.0, 3600000.0), (0, 7200000.0, 3600000.0)]
    actual = model.calls
    assert actual == expected


def test_model_added_after_agents_initializes_them():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.set_space(HomogeneousSpace(dt_agent=(1, 'hour')))
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    agent.set_attribute('cellcycle.mean_length', (10, 'hour'))
    sim.add_agent(agent)
    sim.add_model(TimeBasedCellCycleModel(update_interval=(1, 'hour'), distribution='fixed'))
    sim.run(time=(11, 'hour'), dt=(1, 'hour'), verbosity=0)
    expected = 2
    actual = len(sim.agents)
    assert actual == expected


def test_context_fields_are_read_only():
    from lattics import UpdateContext

    ctx = UpdateContext(time=0, dt=1000)
    with pytest.raises(AttributeError):
        ctx.dt = 5


def test_context_without_simulation_cannot_add_agents():
    from lattics import Agent, UpdateContext

    ctx = UpdateContext(time=0, dt=1000)
    with pytest.raises(RuntimeError):
        ctx.add_agent(Agent())


def test_space_is_created_without_simulation_and_set_once():
    from lattics import HomogeneousSpace, Simulation

    space = HomogeneousSpace()
    Simulation().set_space(space)
    with pytest.raises(RuntimeError):
        Simulation().set_space(space)
