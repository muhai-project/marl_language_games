import pytest
from easydict import EasyDict as edict

from emrl.environment.lexicon import SAPair
from emrl.experiment.experiment import Experiment
from emrl.experiment.monitors import Monitors


@pytest.fixture
def simple_exp():
    cfg = edict()
    exp = Experiment(cfg)
    return exp


def test_init(simple_exp):
    monitors = simple_exp.monitors
    len(monitors.monitors.keys()) == 0


def test_create_monitor(simple_exp):
    monitors = simple_exp.monitors
    monitors.monitors["monitor1"]

    assert len(monitors.monitors) == 1
    assert monitors.monitors["monitor"] == []


def test_add_first_event_to_trial(simple_exp):
    monitors = simple_exp.monitors

    monitor = monitors.monitors["monitor1"]
    monitors.add_event_to_trial(monitor, 0, 0)

    assert len(monitors.monitors["monitor1"]) == 1
    assert monitors.monitors["monitor1"][0] == [0]


def test_add_second_event_to_trial(simple_exp):
    monitors = simple_exp.monitors

    monitor = monitors.monitors["monitor1"]
    monitors.add_event_to_trial(monitor, 0, 0)
    monitors.add_event_to_trial(monitor, 0, 1)

    assert len(monitors.monitors["monitor1"]) == 1
    assert monitors.monitors["monitor1"][0] == [0, 1]


def test_add_second_trial(simple_exp):
    monitors = simple_exp.monitors

    monitor = monitors.monitors["monitor1"]
    monitors.add_event_to_trial(monitor, 0, 0)
    monitors.add_event_to_trial(monitor, 0, 1)
    monitors.add_event_to_trial(monitor, 1, 0)

    assert len(monitors.monitors["monitor1"]) == 2
    assert monitors.monitors["monitor1"][0] == [0, 1]
    assert monitors.monitors["monitor1"][1] == [0]


def test_add_second_monitor(simple_exp):
    monitors = simple_exp.monitors

    monitor = monitors.monitors["monitor1"]
    monitors.add_event_to_trial(monitor, 0, 0)
    monitors.add_event_to_trial(monitor, 0, 1)
    monitors.add_event_to_trial(monitor, 1, 0)
    monitor = monitors.monitors["monitor2"]

    assert len(monitors.monitors) == 2
    assert len(monitors.monitors["monitor1"]) == 2
    assert len(monitors.monitors["monitor2"]) == 0


@pytest.fixture
def exp():
    cfg = edict()
    cfg.ENV = "bng"
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    cfg.CONTEXT_MIN_SIZE = 5
    cfg.CONTEXT_MAX_SIZE = 8
    cfg.EPS_GREEDY = 0
    cfg.INITIAL_Q_VAL = 0.5
    cfg.REWARD_SUCCESS = 0.1
    cfg.REWARD_FAILURE = -0.1
    cfg.EPSILON_FAILURE = 0.01
    cfg.LATERAL_INHIBITION = True
    cfg.UPDATE_RULE = "basic"
    cfg.DELETE_SA_PAIR = False
    cfg.IGNORE_LOW_SA_PAIR = True
    cfg.PRINT_EVERY = 0

    exp = Experiment(cfg)
    exp.initialize()
    exp.env.reset()

    return exp


def test_record_communicative_success(exp):
    exp.env.speaker.lexicon.q_table = [SAPair(exp.env.topic, "f1", initial_value=0.5)]
    exp.env.hearer.lexicon.q_table = [
        SAPair(exp.env.topic, "f1", initial_value=0.5),
    ]
    exp.env.step(0)
    monitors = exp.monitors
    monitors.record_communicative_success(0)
    assert monitors.monitors["communicative-success"] == [[True]]


def test_record_communicative_fail(exp):
    exp.env.speaker.lexicon.q_table = [SAPair(exp.env.topic, "f1", initial_value=0.5)]
    exp.env.hearer.lexicon.q_table = [
        SAPair("not topic", "f1", initial_value=0.5),
    ]
    exp.env.step(0)
    monitors = exp.monitors
    monitors.record_communicative_success(0)
    assert monitors.monitors["communicative-success"] == [[False]]


def test_keep_value_basic():
    cfg = edict()
    cfg.ENV = "bng"
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    cfg.CONTEXT_MIN_SIZE = 5
    cfg.CONTEXT_MAX_SIZE = 8
    cfg.INITIAL_Q_VAL = 0.5
    cfg.REWARD_SUCCESS = 0.1
    cfg.REWARD_FAILURE = -0.1
    cfg.EPSILON_FAILURE = 0.01
    cfg.UPDATE_RULE = "basic"

    exp = Experiment(cfg)
    exp.initialize()

    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=0.5))
    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=1))
    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=0.11))
    assert not exp.monitors.keep_value(SAPair("m1", "f1", initial_value=0.10))
    assert not exp.monitors.keep_value(SAPair("m1", "f1", initial_value=0))


def test_keep_value_int():
    cfg = edict()
    cfg.ENV = "bng"
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    cfg.CONTEXT_MIN_SIZE = 5
    cfg.CONTEXT_MAX_SIZE = 8
    cfg.INITIAL_Q_VAL = 0
    cfg.REWARD_SUCCESS = 1
    cfg.REWARD_FAILURE = -1
    cfg.EPSILON_FAILURE = 0.01
    cfg.UPDATE_RULE = "interpolated"

    exp = Experiment(cfg)
    exp.initialize()

    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=1))
    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=0))
    assert exp.monitors.keep_value(SAPair("m1", "f1", initial_value=-0.99))
    assert not exp.monitors.keep_value(SAPair("m1", "f1", initial_value=-0.991))
    assert not exp.monitors.keep_value(SAPair("m1", "f1", initial_value=-1))


def test_calculate_lexicon_size_count_one(exp):
    exp.env.speaker.lexicon.q_table = [SAPair("m1", "f1", initial_value=0)]
    exp.cfg.IGNORE_LOW_SA_PAIR = False
    monitors = exp.monitors

    size = monitors.calculate_lexicon_size(exp.env.speaker)

    assert size == 1


def test_calculate_lexicon_size_ignore_count_one(exp):
    exp.env.speaker.lexicon.q_table = [SAPair("m1", "f1", initial_value=0.5)]
    exp.cfg.IGNORE_LOW_SA_PAIR = True
    monitors = exp.monitors

    size = monitors.calculate_lexicon_size(exp.env.speaker)

    assert size == 1


def test_calculate_lexicon_size_ignore_not_counted(exp):
    exp.env.speaker.lexicon.q_table = [SAPair("m1", "f1", initial_value=0)]
    exp.cfg.IGNORE_LOW_SA_PAIR = True
    monitors = exp.monitors
    size = monitors.calculate_lexicon_size(exp.env.speaker)
    assert size == 0


def test_calculate_lexicon_size_count_multiple(exp):
    exp.env.speaker.lexicon.q_table = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]
    exp.cfg.IGNORE_LOW_SA_PAIR = False
    monitors = exp.monitors

    size = monitors.calculate_lexicon_size(exp.env.speaker)

    assert size == 3


def test_calculate_lexicon_size_ignore_count_multiple(exp):
    exp.env.speaker.lexicon.q_table = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]
    exp.cfg.IGNORE_LOW_SA_PAIR = True
    monitors = exp.monitors

    size = monitors.calculate_lexicon_size(exp.env.speaker)

    assert size == 2


def test_record_lexicon_size_empty(exp):
    monitors = exp.monitors
    monitors.record_lexicon_size(0)
    assert monitors.monitors["lexicon-size"] == [[0]]


def test_record_lexicon_size_not_empty(exp):
    for i in range(0, 7):
        exp.env.population[i].lexicon.q_table = [
            SAPair("m1", "f1", initial_value=0.5),
            SAPair("m2", "f1", initial_value=0.5),
        ]

    monitors = exp.monitors
    monitors.record_lexicon_size(0)
    assert monitors.monitors["lexicon-size"] == [[1.4]]


def test_lexicon_similarity_same(exp):
    monitors = exp.monitors
    lex1 = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]

    lex2 = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]
    assert monitors.lexicon_similarity(lex1, lex2) == 1


def test_lexicon_similarity_diff(exp):
    monitors = exp.monitors
    lex1 = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]

    lex2 = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
    ]
    assert monitors.lexicon_similarity(lex1, lex2) == 0.8


def test_lexicon_similarity_diff(exp):
    monitors = exp.monitors
    lex1 = [
        SAPair("m1", "f1", initial_value=0.5),
        SAPair("m2", "f1", initial_value=1),
        SAPair("m3", "f1", initial_value=0),
    ]

    lex2 = [
        SAPair("m3", "f8", initial_value=0.5),
        SAPair("m100", "f100", initial_value=1),
    ]
    assert monitors.lexicon_similarity(lex1, lex2) == 0


def test_record_forms_per_meaning_empty(exp):
    monitors = exp.monitors
    monitors.record_forms_per_meaning(0)
    assert monitors.monitors["forms-per-meaning"] == [[0]]


def test_record_forms_per_meaning_not_empty_no_ignore(exp):
    monitors = exp.monitors
    exp.cfg.IGNORE_LOW_SA_PAIR = False
    for i in range(0, 7):
        exp.env.population[i].lexicon.q_table = [
            SAPair("m1", "f1", initial_value=0.5),
            SAPair("m2", "f1", initial_value=0.5),
            SAPair("m2", "f2", initial_value=0.5),
        ]
    monitors.record_forms_per_meaning(0)
    assert monitors.monitors["forms-per-meaning"] == [[1.05]]


def test_record_forms_per_meaning_not_empty_ignore(exp):
    monitors = exp.monitors
    exp.cfg.IGNORE_LOW_SA_PAIR = True
    for i in range(0, 7):
        exp.env.population[i].lexicon.q_table = [
            SAPair("m1", "f1", initial_value=0.5),
            SAPair("m1", "f3", initial_value=0),
            SAPair("m2", "f1", initial_value=0.5),
            SAPair("m2", "f2", initial_value=0.5),
        ]
    monitors.record_forms_per_meaning(0)
    assert monitors.monitors["forms-per-meaning"] == [[1.05]]


def test_record_meanings_per_form_not_empty_ignore(exp):
    monitors = exp.monitors
    exp.cfg.IGNORE_LOW_SA_PAIR = True
    for i in range(0, 7):
        exp.env.population[i].lexicon.q_table = [
            SAPair("m1", "f1", initial_value=0.5),
            SAPair("m1", "f3", initial_value=0),
            SAPair("m2", "f4", initial_value=0.5),
            SAPair("m2", "f2", initial_value=0.5),
        ]
    monitors.record_meanings_per_form(0)
    assert monitors.monitors["meanings-per-form"] == [[0.7]]
