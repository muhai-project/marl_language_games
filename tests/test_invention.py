from emrl.utils.invention import invent, make_id


def test_make_id():
    assert make_id("AG") == "#'AG-0"
    assert make_id("AG") == "#'AG-1"
    assert make_id("OBJ") == "#'OBJ-0"
    assert make_id("AG") == "#'AG-2"
    assert make_id("OBJ") == "#'OBJ-1"


def test_invent():
    f1 = invent(syllables=1)
    assert type(f1) == str and len(f1) == 2
    f2 = invent(syllables=2)
    assert type(f2) == str and len(f2) == 4
    f3 = invent(syllables=3)
    assert type(f3) == str and len(f3) == 6
