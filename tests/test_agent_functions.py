def test_default_instantiation():
    from lattics import Agent

    agent = Agent()
    assert agent is not None


def test_has_attribute():
    from lattics import Agent

    agent = Agent()
    agent.set_attribute('test_attr', 'value')
    expected = True
    actual = agent.has_attribute('test_attr')
    assert actual == expected


def test_get_set_attribute():
    from lattics import Agent

    agent = Agent()
    agent.set_attribute('test_attr', 'value')
    expected = 'value'
    actual = agent.get_attribute('test_attr')
    assert actual == expected


def test_initialize_id():
    from lattics import Agent

    agent_1 = Agent()
    agent_2 = Agent()
    expected = agent_1.id + 1
    actual = agent_2.id
    assert actual == expected


def test_clone():
    from lattics import Agent

    agent_1 = Agent()
    agent_1.set_attribute('test_attr', 'value')
    agent_2 = agent_1.clone()
    assert (not agent_1.id == agent_2.id) and (agent_2.has_attribute('test_attr'))
