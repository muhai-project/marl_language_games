from easydict import EasyDict as edict

from emrl.environment.bng.bng_env import BasicNamingGameEnv
from emrl.environment.lexicon import SAPair


def test_init_world():
    cfg = edict()
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    env = BasicNamingGameEnv(cfg)

    assert len(env.world.objects) == cfg.WORLD_SIZE


def test_init_population():
    cfg = edict()
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    env = BasicNamingGameEnv(cfg)

    assert len(env.population) == cfg.POPULATION_SIZE
    ids = [ag.id for ag in env.population]
    assert len(ids) == len(set(ids))

    lengths = [len(ag.lexicon) for ag in env.population]
    assert all(val == 0 for val in lengths)


def test_reset_roles():
    cfg = edict()
    cfg.WORLD_SIZE = 10
    cfg.POPULATION_SIZE = 10
    cfg.CONTEXT_MIN_SIZE = 5
    cfg.CONTEXT_MAX_SIZE = 8
    env = BasicNamingGameEnv(cfg)

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


def test_step_first():
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

    env.step(0)

    assert env.speaker.id != env.hearer.id
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

    assert env.speaker.lexicon.q_table[0].q_val == cfg.INITIAL_Q_VAL
    assert env.speaker.lexicon.q_table[1].q_val == cfg.INITIAL_Q_VAL

