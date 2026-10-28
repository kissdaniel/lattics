def test_all_public_names_exist():
    import lattics

    missing = [name for name in lattics.__all__ if not hasattr(lattics, name)]
    assert missing == []


def test_names_used_in_documentation_example():
    import lattics

    expected = ['Event', 'HomogeneousSpace', 'Simulation', 'Substrate', 'SubstrateInfo', 'SubstrateNode']
    actual = [name for name in expected if hasattr(lattics, name)]
    assert actual == expected


def test_models_are_exported():
    import lattics

    missing = [name for name in lattics.models.__all__ if not hasattr(lattics.models, name)]
    assert missing == [] and 'MultiPhaseTimeBasedCellCycleModel' in lattics.models.__all__


def test_attribute_names_carry_their_subsystem_prefix():
    from lattics.attributes import known_names

    names = known_names()
    wrong = [name for name in names if name.count('.') != 1 or not all(name.split('.'))]
    assert names and wrong == []


def test_default_of_unknown_attribute_is_missing():
    from lattics.attributes import MISSING, default_of

    assert default_of('my.attribute') is MISSING


def test_helper_imports_are_not_exposed():
    import lattics

    exposed = [name for name in ('Any', 'random') if hasattr(lattics, name)]
    assert exposed == []
