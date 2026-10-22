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
    sim.set_space(HomogeneousSpace(interval=(1, 'hour')))
    model = _recording_model_class()(update_interval=(1, 'hour'))
    sim.add_model(model)
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    sim.add_agent(agent)
    sim.run(time=(3, 'hour'), dt=(30, 'min'), verbosity=0)
    # the model runs every hour, from the start, and advances by one hour each time
    expected = [(0, 0, 3600000.0), (0, 3600000.0, 3600000.0), (0, 7200000.0, 3600000.0)]
    actual = model.calls
    assert actual == expected


def test_model_added_after_agents_initializes_them():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import TimeBasedCellCycleModel

    sim = Simulation()
    sim.set_space(HomogeneousSpace(interval=(1, 'hour')))
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


@pytest.mark.parametrize(
    'division_time, last_update_time, expected',
    [
        (None, 0, False),  # never divided
        (5000, None, False),  # first update of the component
        (5000, 5000, True),  # divided by the space right after the previous update
        (6000, 5000, True),
        (4000, 5000, False),  # already seen at the previous update
    ],
)
def test_divided_since_last_update(division_time, last_update_time, expected):
    from lattics import Agent, UpdateContext

    agent = Agent()
    agent.set_attribute('space.last_division_time', division_time)
    ctx = UpdateContext(time=10000, dt=5000, last_update_time=last_update_time)
    assert ctx.divided_since_last_update(agent) is expected


def test_agent_without_division_time_has_not_divided():
    from lattics import Agent, UpdateContext

    assert UpdateContext(time=10, dt=5, last_update_time=5).divided_since_last_update(Agent()) is False


def test_models_receive_the_time_of_their_previous_update():
    from lattics import Agent, HomogeneousSpace, Simulation
    from lattics.models import BaseModel

    class Recording(BaseModel):
        def __init__(self):
            super().__init__((1, 'hour'))
            self.previous = []

        def update_agent(self, agent, ctx):
            self.previous.append(ctx.last_update_time)

    sim = Simulation()
    sim.set_space(HomogeneousSpace())
    model = Recording()
    sim.add_model(model)
    agent = Agent()
    agent.set_attribute('space.volume', 1)
    sim.add_agent(agent)
    sim.run(time=(2, 'hour'), dt=(1, 'hour'), verbosity=0)
    sim.run(time=(1, 'hour'), dt=(1, 'hour'), verbosity=0)
    assert model.previous == [None, 0, 3600000.0]
