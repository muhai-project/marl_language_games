from emrl.environment.lexicon import Lexicon
from emrl.utils.utils import cfg_from_file

cfg = cfg_from_file("cfg/bng.yml")  # TODO only tests for bng cfg?!
# http://qualityisspeed.blogspot.com/2015/02/the-dependency-elimination-principle-a-canonical-example.html


def test_invent_cxn():
    lex = Lexicon(cfg)
    meaning = "m1"
    assert len(lex) == 0
    new_cxn = lex.invent_cxn(meaning)
    assert len(lex) == 1 and lex.q_table[0] == new_cxn
    assert new_cxn.meaning == meaning
    assert type(new_cxn.form) is str
    assert new_cxn.q_val == cfg.INITIAL_Q_VAL


def test_adopt_cxn():
    lex = Lexicon(cfg)
    meaning1, form1 = "m1", "f1"
    assert len(lex) == 0
    new_cxn1 = lex.adopt_cxn(meaning1, form1)
    assert len(lex) == 1 and lex.q_table[0] == new_cxn1
    assert new_cxn1.meaning == meaning1 and new_cxn1.form == form1
    assert new_cxn1.q_val == cfg.INITIAL_Q_VAL

    meaning2 = "m2", "f2"
    new_cxn2 = lex.adopt_cxn(meaning2, form1)
    assert len(lex) == 2
    assert lex.q_table[0] == new_cxn1
    assert lex.q_table[1] == new_cxn2


def test_cxns_with_meaning():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    for meaning, form in zip(meanings, forms):
        lex.adopt_cxn(meaning, form)
    assert len(lex) == 5
    filtered_cxns = lex.get_cxns_with_meaning("m1")
    assert (
        len(filtered_cxns) == 1
        and filtered_cxns[0].meaning == "m1"
        and filtered_cxns[0].form == "f1"
    )
    lex.adopt_cxn("m1", "f2")
    filtered_cxns = lex.get_cxns_with_meaning("m1")
    assert len(filtered_cxns) == 2
    for cxn in filtered_cxns:
        assert cxn.meaning == "m1"


def test_remove_cxn():
    lex = Lexicon(cfg)
    meanings = ["m1", "m2", "m3", "m4", "m5"]
    forms = ["f1", "f2", "f3", "f4", "f5"]
    last = None
    for meaning, form in zip(meanings, forms):
        last = lex.adopt_cxn(meaning, form)
    assert len(lex) == 5
    lex.remove_cxn(last)
    assert len(lex) == 4
