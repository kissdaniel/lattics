def test_all_public_names_exist():
    import lattics

    missing = [name for name in lattics.__all__ if not hasattr(lattics, name)]
    assert missing == []


def test_names_used_in_documentation_example():
    import lattics

    expected = ['Event', 'HomogeneousSpace', 'Simulation', 'SubstrateInfo', 'SubstrateNode']
    actual = [name for name in expected if hasattr(lattics, name)]
    assert actual == expected


def test_models_are_exported():
    import lattics

    missing = [name for name in lattics.models.__all__ if not hasattr(lattics.models, name)]
    assert missing == [] and 'MultiPhaseTimeBasedCellCycleModel' in lattics.models.__all__


def test_helper_imports_are_not_exposed():
    import lattics

    exposed = [name for name in ('Any', 'random') if hasattr(lattics, name)]
    assert exposed == []
