from marl_language_games.utils.invention import ids, invent, make_id


def test_make_id():
    ag_id = ids["AGENT"]
    obj_id = ids["OBJECT"]

    assert make_id("AGENT") == f"#'AGENT-{ag_id+0}"
    assert make_id("AGENT") == f"#'AGENT-{ag_id+1}"
    assert make_id("OBJECT") == f"#'OBJECT-{obj_id+0}"
    assert make_id("AGENT") == f"#'AGENT-{ag_id+2}"
    assert make_id("OBJECT") == f"#'OBJECT-{obj_id+1}"


def test_invent():
    f1 = invent(syllables=1)
    assert type(f1) == str and len(f1) == 2
    f2 = invent(syllables=2)
    assert type(f2) == str and len(f2) == 4
    f3 = invent(syllables=3)
    assert type(f3) == str and len(f3) == 6
