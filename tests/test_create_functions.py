import pytest


def test_create_simulation_2d_converts_size_to_lattice_sites():
    import lattics

    sim = lattics.create_simulation_2d(dimensions=((1, 'mm'), (400, 'um')), grid_spacing=20)
    expected = (50, 20)
    actual = sim._space._agent_layer.shape
    assert actual == expected


def test_create_simulation_2d_rejects_size_not_multiple_of_grid_spacing():
    import lattics

    with pytest.raises(ValueError):
        lattics.create_simulation_2d(dimensions=((750, 'um'), (750, 'um')), grid_spacing=20)


def test_create_simulation_2d_requires_size():
    import lattics

    with pytest.raises(ValueError):
        lattics.create_simulation_2d()
