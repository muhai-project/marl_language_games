from marl_language_games.utils.convert_data import convert_monitor


def test_convert_monitor_int():
    monitor = [[0, 1, 2, 3], [4, 5, 6, 7]]
    data = convert_monitor(monitor)

    assert data == "(((0 1 2 3)(4 5 6 7)))"


def test_convert_monitor_bool():
    monitor = [[True, False, True], [False, False, False]]
    data = convert_monitor(monitor)

    assert data == "(((1 0 1)(0 0 0)))"


def test_convert_monitor_float():
    monitor = [[0.1, 1.0102012, 2.8888], [4.5, 5.9999, 6.0]]
    data = convert_monitor(monitor)

    assert data == "(((0.1 1.01 2.89)(4.5 6.0 6.0)))"
