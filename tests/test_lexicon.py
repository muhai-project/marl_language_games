from emrl.environment.lexicon import Lexicon
from emrl.utils.cfg import cfg_from_file

cfg = cfg_from_file("cfg/bng.yml")  # TODO only tests for bng cfg?!
# http://qualityisspeed.blogspot.com/2015/02/the-dependency-elimination-principle-a-canonical-example.html


def test_invent_sa_pair():
    lex = Lexicon(cfg)
    meaning = "m1"
    assert len(lex) == 0
    new_sa_pair = lex.invent_sa_pair(meaning)
    assert len(lex) == 1 and lex.q_table[0] == new_sa_pair
    assert new_sa_pair.meaning == meaning
    assert type(new_sa_pair.form) is str
    assert new_sa_pair.q_val == cfg.INITIAL_Q_VAL


def test_adopt_sa_pair():
    lex = Lexicon(cfg)
    meaning1, form1 = "m1", "f1"
    assert len(lex) == 0
    new_sa_pair1 = lex.adopt_sa_pair(meaning1, form1)
    assert len(lex) == 1 and lex.q_table[0] == new_sa_pair1
    assert new_sa_pair1.meaning == meaning1 and new_sa_pair1.form == form1
    assert new_sa_pair1.q_val == cfg.INITIAL_Q_VAL

    meaning2 = "m2", "f2"
    new_sa_pair2 = lex.adopt_sa_pair(meaning2, form1)
    assert len(lex) == 2
    assert lex.q_table[0] == new_sa_pair1
    assert lex.q_table[1] == new_sa_pair2


def test_get_actions_produce():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    for meaning, form in zip(meanings, forms):
        lex.adopt_sa_pair(meaning, form)
    assert len(lex) == 5
    filtered_sa_pairs = lex.get_actions_produce("m1")
    assert (
        len(filtered_sa_pairs) == 1
        and filtered_sa_pairs[0].meaning == "m1"
        and filtered_sa_pairs[0].form == "f1"
    )
    lex.adopt_sa_pair("m1", "f2")
    filtered_sa_pairs = lex.get_actions_produce("m1")
    assert len(filtered_sa_pairs) == 2
    for sa_pair in filtered_sa_pairs:
        assert sa_pair.meaning == "m1"
        assert sa_pair.form == "f1" or sa_pair.form == "f2"


def test_get_actions_comprehend():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    for meaning, form in zip(meanings, forms):
        lex.adopt_sa_pair(meaning, form)
    assert len(lex) == 5
    filtered_sa_pairs = lex.get_actions_comprehend("f2")
    assert (
        len(filtered_sa_pairs) == 1
        and filtered_sa_pairs[0].meaning == "m2"
        and filtered_sa_pairs[0].form == "f2"
    )
    lex.adopt_sa_pair("m1", "f2")
    filtered_sa_pairs = lex.get_actions_comprehend("f2")
    assert len(filtered_sa_pairs) == 2
    for sa_pair in filtered_sa_pairs:
        assert sa_pair.form == "f2"
        assert sa_pair.meaning == "m1" or sa_pair.meaning == "m2"


def test_remove_sa_pair():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    last = None
    for meaning, form in zip(meanings, forms):
        last = lex.adopt_sa_pair(meaning, form)
    assert len(lex) == 5
    assert last in lex.q_table
    lex.remove_sa_pair(last)
    assert len(lex) == 4
    assert last not in lex.q_table


def test_lex_repr():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    for meaning, form in zip(meanings, forms):
        lex.adopt_sa_pair(meaning, form)
    lex_repr = str(lex)

    for meaning in meanings:
        assert meaning in lex_repr
    for form in forms:
        assert form in lex_repr

    first = lex.q_table[0]
    first.q_val = 500.0156
    assert lex.q_table[0].q_val == 500.0156
    lex_repr = str(lex)
    assert "500.016" in lex_repr
