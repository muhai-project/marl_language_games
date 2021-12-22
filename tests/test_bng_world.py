import pytest

from emrl.environment.bng.bng_env import World


def test_init():
    world = World(5)
    assert len(world.objects) == 5
    world = World(9)
    assert len(world.objects) == 9


def test_pick_context_content():
    world = World(10)
    context = world.pick_context(5, 5)
    assert len(context) == 5
    assert set(context) == set(world.objects).intersection(set(context))


@pytest.mark.parametrize("min_size, max_size", [(6, 9), (1, 2), (5, 10)])
def test_pick_context_return(min_size, max_size):
    world = World(10)
    context = world.pick_context(min_size, max_size)
    assert min_size <= len(context) <= max_size


@pytest.mark.parametrize("context", [[100], [1, 2, 3], [-1, 1, 5, 10, 9]])
def test_pick_topic(context):
    world = World(10)
    topic = world.pick_topic(context)
    assert topic in context
