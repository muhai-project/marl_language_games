import pytest
from easydict import EasyDict as edict

from marl_language_games.environment.bng.bng_env import BasicNamingGameEnv
from marl_language_games.environment.lexicon import SAPair


@pytest.fixture
def simple_env():
    cfg = edict()
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    cfg.CONTEXT_MIN_SIZE = 5
    cfg.CONTEXT_MAX_SIZE = 8
    env = BasicNamingGameEnv(cfg)
    return env, cfg


def test_init_world(simple_env):
    env, cfg = simple_env

    assert len(env.world.objects) == cfg.WORLD_SIZE


def test_init_population(simple_env):
    env, cfg = simple_env

    ids = [ag.id for ag in env.population]

    assert len(env.population) == cfg.POPULATION_SIZE
    assert len(ids) == len(set(ids))
    assert all(val == 0 for val in [len(ag.lexicon) for ag in env.population])


def test_reset_roles(simple_env):
    env, _ = simple_env
    env.reset()

    assert env.speaker in env.population
    assert env.hearer in env.population
    assert env.speaker.id != env.hearer.id

    assert 5 <= len(env.context) <= 8
    assert all(obj in env.world.objects for obj in env.context)
    assert env.topic in env.context

    assert env.speaker.context == env.context
    assert env.hearer.context == env.context

    assert env.lexicon_change is False
    assert env.lexicon_coherence is False


@pytest.fixture
def environment_and_cfg():
    cfg = edict()
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

    env = BasicNamingGameEnv(cfg)
    env.reset()
    return env, cfg


def test_logging_success(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.cfg.PRINT_EVERY = 1
    env.step(0)


def test_step_first(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert len(env.speaker.lexicon) == 1
    assert len(env.hearer.lexicon) == 1

    assert env.topic == env.speaker.lexicon.q_table[0].meaning
    assert env.topic in env.context
    assert env.speaker.context == env.context
    assert env.hearer.context == env.context

    assert env.speaker.lexicon.q_table[0].q_val == 0.4
    assert env.hearer.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL


def test_step_no_invention_then_adoption(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert set(env.speaker.lexicon.q_table) == set(env.hearer.lexicon.q_table)
    assert len(env.hearer.lexicon) == 1
    assert env.hearer.lexicon.q_table[0].q_val == 0.5
    assert env.speaker.lexicon.q_table[0].q_val == 0.4


def test_step_no_invention_no_adoption_no_success(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.hearer.lexicon.q_table = [
        SAPair("not topic", "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert env.speaker.lexicon.q_table[0].meaning == env.topic
    assert env.speaker.lexicon.q_table[0].form == "f1"
    assert env.speaker.lexicon.q_table[0].q_val == 0.4

    assert len(env.hearer.lexicon) == 2
    assert env.hearer.lexicon.q_table[0].meaning == "not topic"
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.4

    assert env.hearer.lexicon.q_table[1].meaning == env.topic
    assert env.hearer.lexicon.q_table[1].form == "f1"
    assert env.hearer.lexicon.q_table[1].q_val == 0.5


def test_step_no_invention_no_adoption_success(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.hearer.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.step(0)

    assert env.lexicon_change is False
    assert env.lexicon_coherence is True
    assert env.speaker.communicative_success is True
    assert env.hearer.communicative_success is True

    assert env.speaker.lexicon.q_table[0].meaning == env.topic
    assert env.speaker.lexicon.q_table[0].form == "f1"
    assert env.speaker.lexicon.q_table[0].q_val == 0.6

    assert len(env.hearer.lexicon) == 1
    assert env.hearer.lexicon.q_table[0].meaning == env.topic
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.6


def test_step_no_coherence_success(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.hearer.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=0.5),
        SAPair(env.topic, "f2", initial_value=1),
    ]
    env.step(0)

    assert env.lexicon_change is False
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is True
    assert env.hearer.communicative_success is True

    assert len(env.speaker.lexicon) == 1
    assert env.speaker.lexicon.q_table[0].meaning == env.topic
    assert env.speaker.lexicon.q_table[0].form == "f1"
    assert env.speaker.lexicon.q_table[0].q_val == 0.6

    assert len(env.hearer.lexicon) == 2
    assert env.hearer.lexicon.q_table[0].meaning == env.topic
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.6

    assert env.hearer.lexicon.q_table[1].meaning == env.topic
    assert env.hearer.lexicon.q_table[1].form == "f2"
    assert env.hearer.lexicon.q_table[1].q_val == 0.9


def test_step_sa_pair_zero_score(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.hearer.lexicon.q_table = [
        SAPair("not topic", "f1", initial_value=0.1),
    ]
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert env.speaker.lexicon.q_table[0].meaning == env.topic
    assert env.speaker.lexicon.q_table[0].form == "f1"
    assert env.speaker.lexicon.q_table[0].q_val == 0.4

    assert len(env.hearer.lexicon) == 2
    assert env.hearer.lexicon.q_table[0].meaning == "not topic"
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0

    assert env.hearer.lexicon.q_table[1].meaning == env.topic
    assert env.hearer.lexicon.q_table[1].form == "f1"
    assert env.hearer.lexicon.q_table[1].q_val == 0.5


def test_step_sa_pair_deletion(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.hearer.cfg.DELETE_SA_PAIR = True
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.hearer.lexicon.q_table = [
        SAPair("not topic", "f1", initial_value=0.1),
    ]
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert env.speaker.lexicon.q_table[0].meaning == env.topic
    assert env.speaker.lexicon.q_table[0].form == "f1"
    assert env.speaker.lexicon.q_table[0].q_val == 0.4

    assert len(env.hearer.lexicon) == 1

    assert env.hearer.lexicon.q_table[0].meaning == env.topic
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.5


def test_step_both_roles_delete(environment_and_cfg):
    env, _ = environment_and_cfg
    env.speaker.cfg.DELETE_SA_PAIR = True
    env.hearer.cfg.DELETE_SA_PAIR = True
    env.speaker.lexicon.q_table = [SAPair(env.topic, "f1", initial_value=0.1)]
    env.hearer.lexicon.q_table = [
        SAPair("not topic", "f1", initial_value=0.1),
    ]
    env.step(0)

    assert env.lexicon_change is True
    assert env.lexicon_coherence is False
    assert env.speaker.communicative_success is False
    assert env.hearer.communicative_success is False

    assert len(env.speaker.lexicon) == 0
    assert len(env.hearer.lexicon) == 1
    assert env.hearer.lexicon.q_table[0].meaning == env.topic
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.5


def test_step_lateral_inhibition_speaker(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f2", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f3", initial_value=cfg.INITIAL_Q_VAL),
        SAPair("not topic", "f1", initial_value=cfg.INITIAL_Q_VAL),
    ]
    env.hearer.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.step(0)

    assert env.lexicon_change is False
    assert env.lexicon_coherence is True
    assert env.speaker.communicative_success is True
    assert env.hearer.communicative_success is True

    assert len(env.speaker.lexicon) == 4
    assert env.speaker.lexicon.q_table[0].q_val == 0.6
    assert env.speaker.lexicon.q_table[1].q_val == 0.4
    assert env.speaker.lexicon.q_table[2].q_val == 0.4
    assert env.speaker.lexicon.q_table[3].q_val == 0.5

    assert len(env.hearer.lexicon) == 1
    assert env.hearer.lexicon.q_table[0].meaning == env.topic
    assert env.hearer.lexicon.q_table[0].form == "f1"
    assert env.hearer.lexicon.q_table[0].q_val == 0.6


def test_step_lateral_inhibition_hearer(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL),
    ]
    env.hearer.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f2", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f3", initial_value=cfg.INITIAL_Q_VAL),
        SAPair("not topic", "f1", initial_value=cfg.INITIAL_Q_VAL),
    ]
    env.step(0)

    assert env.lexicon_change is False
    assert env.lexicon_coherence is True
    assert env.speaker.communicative_success is True
    assert env.hearer.communicative_success is True

    assert len(env.speaker.lexicon) == 1
    assert env.speaker.lexicon.q_table[0].q_val == 0.6

    assert len(env.hearer.lexicon) == 4
    assert env.hearer.lexicon.q_table[0].q_val == 0.6
    assert env.hearer.lexicon.q_table[1].q_val == 0.4
    assert env.hearer.lexicon.q_table[2].q_val == 0.4
    assert env.hearer.lexicon.q_table[3].q_val == 0.5


def test_step_disable_lateral_inhibition(environment_and_cfg):
    env, cfg = environment_and_cfg
    env.speaker.cfg.LATERAL_INHIBITION = False
    env.speaker.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f2", initial_value=cfg.INITIAL_Q_VAL),
        SAPair(env.topic, "f3", initial_value=cfg.INITIAL_Q_VAL),
        SAPair("not topic", "f1", initial_value=cfg.INITIAL_Q_VAL),
    ]
    env.hearer.lexicon.q_table = [
        SAPair(env.topic, "f1", initial_value=cfg.INITIAL_Q_VAL)
    ]
    env.step(0)

    assert env.lexicon_change is False
    assert env.lexicon_coherence is True
    assert env.speaker.communicative_success is True
    assert env.hearer.communicative_success is True

    assert len(env.speaker.lexicon) == 4
    assert env.speaker.lexicon.q_table[0].q_val == 0.6
    assert env.speaker.lexicon.q_table[1].q_val == 0.5
    assert env.speaker.lexicon.q_table[2].q_val == 0.5
    assert env.speaker.lexicon.q_table[3].q_val == 0.5
