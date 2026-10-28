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


def test_new_agent_has_no_id():
    from lattics import Agent

    agent = Agent()
    assert agent.id is None


def test_clone():
    from lattics import Agent

    agent_1 = Agent()
    agent_1.set_attribute('test_attr', 'value')
    agent_2 = agent_1.clone()
    agent_2.set_attribute('test_attr', 'changed')
    assert agent_2.id is None and agent_1.get_attribute('test_attr') == 'value'
