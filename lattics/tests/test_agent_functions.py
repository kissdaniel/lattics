def test_default_instantiation():
    from lattics.core._agent import Agent

    agent = Agent()
    assert agent is not None


def test_has_attribute():
    from lattics.core._agent import Agent

    agent = Agent()
    agent.set_attribute('test_attr', 'value')
    expected = True
    actual = agent.has_attribute('test_attr')
    assert actual == expected


def test_get_set_attribute():
    from lattics.core._agent import Agent

    agent = Agent()
    agent.set_attribute('test_attr', 'value')
    expected = 'value'
    actual = agent.get_attribute('test_attr')
    assert actual == expected


def test_initialize_id():
    from lattics.core._agent import Agent

    agent_1 = Agent()
    agent_2 = Agent()
    expected_1 = 0
    expected_2 = 1
    actual_1 = agent_1.id
    actual_2 = agent_2.id
    assert actual_1 == expected_1 and actual_2 == expected_2


def test_clone():
    from lattics.core._agent import Agent

    agent_1 = Agent()
    agent_1.set_attribute('test_attr', 'value')
    agent_2 = agent_1.clone()
    assert (not agent_1.id == agent_2.id) and (agent_2.has_attribute('test_attr'))
