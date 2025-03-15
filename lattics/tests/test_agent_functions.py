def test_default_instantiation():
    from lattics import agent
    agent = agent.Agent()
    assert agent is not None


def test_status_flag_initialization_get_set_bool():
    from lattics import agent
    agent = agent.Agent()
    agent.initialize_status_flag('test_flag_str')
    expected = True
    agent.set_status_flag('test_flag_str', expected)
    actual = agent.get_status_flag('test_flag_str')
    assert actual == expected


def test_status_flag_initialization_get_set_str():
    from lattics import agent
    agent = agent.Agent()
    agent.initialize_status_flag('test_flag_str')
    expected = 'test_value'
    agent.set_status_flag('test_flag_str', expected)
    actual = agent.get_status_flag('test_flag_str')
    assert actual == expected