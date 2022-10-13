import random

import pytest
from easydict import EasyDict as edict

from marl_language_games.environment.bng.bng_agent import HEARER, SPEAKER, Agent
from marl_language_games.environment.lexicon import SAPair

DUMMY = ""


def test_empty_lexicon():
    cfg = edict()
    agent = Agent(cfg)
    assert len(agent.lexicon) == 0


def test_reset():
    cfg = edict()
    agent = Agent(cfg)
    agent.applied_sa_pair = DUMMY
    agent.communicative_success = False
    agent.context = [DUMMY, DUMMY]

    agent.reset(context=[1, 2, 3])
    assert agent.communicative_success is True
    assert agent.applied_sa_pair is None
    assert agent.context == [1, 2, 3]


@pytest.mark.parametrize(
    "topic, utterance", [("m1", "f1"), ("----", "111"), ("m3", "f3")]
)
def test_adopt_one_sa_pair(topic, utterance):
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    agent = Agent(cfg)
    agent.adopt(topic, utterance)
    assert len(agent.lexicon) == 1
    assert agent.lexicon.q_table[0].meaning == topic
    assert agent.lexicon.q_table[0].form == utterance
    assert agent.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL


def test_adopt_repeat():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    topic, utterance = "m1", "f1"
    agent = Agent(cfg)
    agent.adopt(topic, utterance)
    assert len(agent.lexicon) == 1
    agent.adopt(topic, utterance)
    assert len(agent.lexicon) == 1


def test_adopt_multiple_sa_pairs():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    agent = Agent(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    for topic, utterance in zip(meanings, forms):
        agent.adopt(topic, utterance)

    assert len(agent.lexicon) == 5
    for idx, (topic, utterance) in enumerate(zip(meanings, forms), 0):
        assert agent.lexicon.q_table[idx].meaning == topic
        assert agent.lexicon.q_table[idx].form == utterance
        assert agent.lexicon.q_table[idx].q_val == cfg.INITIAL_Q_VAL


def test_epsilon_greedy_exploit():
    cfg = edict()
    agent = Agent(cfg)

    actions = [SAPair(i, i, initial_value=i) for i in range(5)]
    random.shuffle(actions)
    best_action = agent.epsilon_greedy(actions, eps=0)
    assert SAPair(4, 4) == best_action
    assert best_action.q_val == 4


def test_epsilon_greedy_exploit_equal():
    cfg = edict()
    agent = Agent(cfg)

    actions = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 100),
        SAPair("m5", "f2", 2),
        SAPair("m1", "f3", 101),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f3", 3),
    ]
    random.shuffle(actions)
    best_action = agent.epsilon_greedy(actions, eps=0)
    assert SAPair("m1", "f3") == best_action
    assert best_action.q_val == 101


def test_find_in_context_empty():
    cfg = edict()
    agent = Agent(cfg)
    agent.context = ["m10", "m11", "m12"]
    actions = [SAPair("m1", "f1"), SAPair("m9", "f9"), SAPair("m3", "f3")]
    found = agent.find_in_context(actions)
    assert len(found) == 0


def test_find_in_context_subset():
    cfg = edict()
    agent = Agent(cfg)
    agent.context = ["m1", "m5", "m9"]
    actions = [SAPair("m1", "f1"), SAPair("m9", "f9"), SAPair("m3", "f3")]
    found = agent.find_in_context(actions)
    assert set(found) == set([SAPair("m9", "f9"), SAPair("m1", "f1")])
    assert len(found) == 2


def test_find_in_context_full():
    cfg = edict()
    agent = Agent(cfg)
    agent.context = ["m1", "m5", "m9"]
    actions = [SAPair("m1", "f1"), SAPair("m9", "f9"), SAPair("m5", "f5")]
    found = agent.find_in_context(actions)
    assert set(found) == set(
        [SAPair("m9", "f9"), SAPair("m1", "f1"), SAPair("m5", "f5")]
    )
    assert len(found) == 3


def test_policy_speaker_invent():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    agent = Agent(cfg)
    assert len(agent.lexicon) == 0
    form, invented = agent.policy(SPEAKER, "m1")
    assert len(agent.lexicon) == 1
    assert agent.lexicon.q_table[0].form == form
    assert agent.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL
    assert invented


def test_policy_speaker_exploit():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 3),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f3", 3),
    ]
    form, invented = agent.policy(SPEAKER, "m5")
    assert form == "f3"
    assert invented is False


def test_policy_hearer_comprehend_none():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    assert len(agent.lexicon) == 0
    meaning = agent.policy(HEARER, DUMMY)
    assert meaning is None


def test_policy_hearer_comprehend_exploit():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f1", 3),
    ]
    meaning = agent.policy(HEARER, "f1")
    assert meaning == "m5"
    assert agent.applied_sa_pair == SAPair("m5", "f1")
    assert agent.applied_sa_pair.q_val == 6


def test_produce_as_hearer_empty_with_context():
    cfg = edict()
    agent = Agent(cfg)
    agent.context = ["m1", "m6"]
    assert len(agent.lexicon) == 0
    form = agent.produce_as_hearer("m6")
    assert form is None


def test_produce_as_hearer_empty_with_no_context():
    cfg = edict()
    agent = Agent(cfg)
    agent.context = ["m1", "m6"]
    assert len(agent.lexicon) == 0
    form = agent.produce_as_hearer("m3")
    assert form is None


def test_produce_as_hearer_context_no_match():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    agent.context = ["m1", "m2", "m3"]
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f1", 3),
    ]
    form = agent.produce_as_hearer("m5")
    assert form is None


def test_produce_as_hearer():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    agent.context = ["m3", "m5", "m4"]
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f1", 3),
    ]
    form = agent.produce_as_hearer("m5")
    assert form == "f3"
    form = agent.produce_as_hearer("m3")
    assert form == "f4"
    form = agent.produce_as_hearer("m4")
    assert form is None


def test_produce_as_hearer_not_in_context():
    cfg = edict()
    cfg.EPS_GREEDY = 0
    agent = Agent(cfg)
    agent.context = ["m3", "m5", "m4"]
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
    ]
    form = agent.produce_as_hearer("m1")
    assert form is None


def test_remove_sa_pair_disabled():
    cfg = edict()
    cfg.DELETE_SA_PAIR = False
    agent = Agent(cfg)
    agent.context = ["m3", "m5", "m4"]
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f1", 3),
    ]
    assert len(agent.lexicon) == 6
    agent.remove_sa_pair(SAPair("m5", "f2"))
    assert len(agent.lexicon) == 6


def test_remove_sa_pair():
    cfg = edict()
    cfg.DELETE_SA_PAIR = True
    agent = Agent(cfg)

    agent.context = ["m3", "m5", "m4"]
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 1),
        SAPair("m5", "f1", 6),
        SAPair("m5", "f2", 2),
        SAPair("m5", "f3", 100),
        SAPair("m3", "f4", 6),
        SAPair("m3", "f1", 3),
    ]
    assert len(agent.lexicon) == 6
    agent.remove_sa_pair(SAPair("m5", "f2"))
    assert len(agent.lexicon) == 5
    for sa_pair in agent.lexicon.q_table:
        assert sa_pair != SAPair("m5", "f2")


def test_update_invalid_rule():
    cfg = edict()
    cfg.UPDATE_RULE = "inter"
    agent = Agent(cfg)
    with pytest.raises(ValueError):
        agent.update(SAPair("m1", "f1"), 1)


def test_update_int_only_update_one():
    cfg = edict()
    cfg.UPDATE_RULE = "interpolated"
    cfg.INITIAL_Q_VAL = 0
    cfg.LEARNING_RATE = 0.1
    cfg.REWARD_FAILURE = -1
    cfg.EPSILON_FAILURE = 0.01
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m2", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m3", "f2", cfg.INITIAL_Q_VAL),
    ]
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    agent.update(agent.lexicon.q_table[1], reward=1)
    assert agent.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == 0.1
    assert agent.lexicon.q_table[2].q_val == cfg.INITIAL_Q_VAL


def test_update_int_repeated():
    cfg = edict()
    cfg.UPDATE_RULE = "interpolated"
    cfg.LEARNING_RATE = 0.1
    cfg.REWARD_FAILURE = -1
    cfg.EPSILON_FAILURE = 0.01
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 0),
        SAPair("m2", "f1", 0),
        SAPair("m3", "f2", 0),
    ]
    assert agent.lexicon.q_table[1].q_val == 0
    agent.update(agent.lexicon.q_table[1], reward=1)
    assert agent.lexicon.q_table[1].q_val == 0.1
    agent.update(agent.lexicon.q_table[1], reward=1)
    assert agent.lexicon.q_table[1].q_val == 0.19
    agent.update(agent.lexicon.q_table[1], reward=-1)
    assert round(agent.lexicon.q_table[1].q_val, 3) == 0.071


def test_update_int_deletion():
    cfg = edict()
    cfg.UPDATE_RULE = "interpolated"
    cfg.INITIAL_Q_VAL = 0
    cfg.LEARNING_RATE = 0.5
    cfg.REWARD_FAILURE = -1
    cfg.EPSILON_FAILURE = 0.01
    cfg.DELETE_SA_PAIR = True
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m2", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m3", "f2", cfg.INITIAL_Q_VAL),
    ]
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    for i in range(50):
        agent.update(agent.lexicon.q_table[1], reward=-1)
        if len(agent.lexicon) == 2:
            break
    assert len(agent.lexicon) == 2


def test_update_basic_only_update_one():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    cfg.UPDATE_RULE = "basic"
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 0.5),
        SAPair("m2", "f1", 0.5),
        SAPair("m3", "f2", 0.5),
    ]
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    agent.update(agent.lexicon.q_table[1], reward=0.3)
    assert agent.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL
    assert agent.lexicon.q_table[1].q_val == 0.8
    assert agent.lexicon.q_table[2].q_val == cfg.INITIAL_Q_VAL


def test_update_basic_repeated_up():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    cfg.UPDATE_RULE = "basic"
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m2", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m3", "f2", cfg.INITIAL_Q_VAL),
    ]
    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    agent.update(agent.lexicon.q_table[1], reward=0.25)
    assert agent.lexicon.q_table[1].q_val == 0.75
    agent.update(agent.lexicon.q_table[1], reward=0.25)
    assert agent.lexicon.q_table[1].q_val == 1
    agent.update(agent.lexicon.q_table[1], reward=0.25)
    assert agent.lexicon.q_table[1].q_val == 1


def test_update_basic_deletion():
    cfg = edict()
    cfg.INITIAL_Q_VAL = 0.5
    cfg.UPDATE_RULE = "basic"
    cfg.DELETE_SA_PAIR = True
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m2", "f1", cfg.INITIAL_Q_VAL),
        SAPair("m3", "f2", cfg.INITIAL_Q_VAL),
    ]

    assert agent.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL
    for i in range(5):
        agent.update(agent.lexicon.q_table[1], reward=-0.25)
        if len(agent.lexicon) == 2:
            break
    assert len(agent.lexicon) == 2
    for sa_pair in agent.lexicon.q_table:
        assert sa_pair.q_val == cfg.INITIAL_Q_VAL


def test_align_LI_no_competitors():
    cfg = edict()
    cfg.UPDATE_RULE = "basic"
    cfg.DELETE_SA_PAIR = True
    cfg.REWARD_SUCCESS = 0.1
    cfg.REWARD_FAILURE = -0.1
    cfg.LATERAL_INHIBITION = True
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 0.5),
        SAPair("m2", "f2", 0.5),
        SAPair("m1", "f3", 0.6),
    ]

    agent.applied_sa_pair = agent.lexicon.q_table[1]
    agent.communicative_success = True

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.5
    assert agent.lexicon.q_table[2].q_val == 0.6

    agent.align()

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.6
    assert agent.lexicon.q_table[2].q_val == 0.6


def test_align_LI_competitors():
    cfg = edict()
    cfg.UPDATE_RULE = "basic"
    cfg.DELETE_SA_PAIR = True
    cfg.REWARD_SUCCESS = 0.1
    cfg.REWARD_FAILURE = -0.1
    cfg.LATERAL_INHIBITION = True
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 0.5),
        SAPair("m2", "f2", 0.5),
        SAPair("m1", "f3", 0.6),
        SAPair("m2", "f5", 0.9),
    ]

    agent.applied_sa_pair = agent.lexicon.q_table[1]
    agent.communicative_success = True

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.5
    assert agent.lexicon.q_table[2].q_val == 0.6
    assert agent.lexicon.q_table[3].q_val == 0.9

    agent.align()

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.6
    assert agent.lexicon.q_table[2].q_val == 0.6
    assert agent.lexicon.q_table[3].q_val == 0.8


def test_align_fail():
    cfg = edict()
    cfg.UPDATE_RULE = "basic"
    cfg.DELETE_SA_PAIR = True
    cfg.REWARD_FAILURE = -0.1
    agent = Agent(cfg)
    agent.lexicon.q_table = [
        SAPair("m1", "f1", 0.5),
        SAPair("m2", "f2", 0.5),
        SAPair("m1", "f3", 0.6),
        SAPair("m2", "f5", 0.9),
    ]

    agent.applied_sa_pair = agent.lexicon.q_table[1]
    agent.communicative_success = False

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.5
    assert agent.lexicon.q_table[2].q_val == 0.6
    assert agent.lexicon.q_table[3].q_val == 0.9

    agent.align()

    assert agent.lexicon.q_table[0].q_val == 0.5
    assert agent.lexicon.q_table[1].q_val == 0.4
    assert agent.lexicon.q_table[2].q_val == 0.6
    assert agent.lexicon.q_table[3].q_val == 0.9
