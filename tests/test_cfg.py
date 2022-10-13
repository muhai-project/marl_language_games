from marl_language_games.utils.cfg import cfg_from_file


def test_bng_cfg():
    cfg = cfg_from_file("cfg/bng.yml")
    assert cfg.ENV == "bng"

    assert cfg.TRIALS >= 1
    assert cfg.EPISODES >= 1
    assert cfg.CONTEXT_MIN_SIZE >= 2
    assert cfg.CONTEXT_MAX_SIZE <= cfg.WORLD_SIZE
    assert cfg.WORLD_SIZE >= 2
    assert cfg.POPULATION_SIZE >= 2

    assert cfg.UPDATE_RULE == "interpolated"
    assert cfg.LEARNING_RATE >= 0 and cfg.LEARNING_RATE <= 1
    assert cfg.EPS_GREEDY >= 0 and cfg.EPS_GREEDY <= 1

    assert cfg.REWARD_SUCCESS > cfg.REWARD_FAILURE
    # epsilon should be small, so rather ~ 0.01 < e 0.1
    assert cfg.EPSILON_FAILURE <= 0.1
