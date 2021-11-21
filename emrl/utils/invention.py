from collections import defaultdict

import numpy as np

ids = defaultdict(int)


def make_id(name):
    """
    Creates a unique id for an object resembling a lisp symbol.

    Each input gets its own counter that is incremented each time the function is called.
    To use:
    >>> make_id("AG")
    #'AG-0
    >>> make_id("AG")
    #'AG-1
    >>> make_id("OBJ")
    #'OBJ-0
    >>> make_id("AG")
    #'AG-2
    >>> make_id("OBJ")
    #'OBJ-1

    Args:
        name: A string for which a so-called symbolic id is created.

    Returns:
        A string that is unique each time the function is called.
    """
    global ids
    val = f"#'{name}-{ids[name]}"
    ids[name] += 1
    return val


def invent(syllables=3):
    """
    Invents a word with a number of syllables through random sampling of letters.

    Each syllable has exactly two letters: a consonant and a vowel (in that order).
    The two letters are composed by randomly sampling from the set of possible choices.

    Args:
        syllables: An integer representing the amount of syllables in the new word.

    Returns:
        A string that is randomly generated.
    """
    vowels = ["a", "e", "i", "o", "u"]
    consonants = [
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    # produce word containing a sequence of syllables
    new_word = ""
    for i in range(syllables):
        new_word += np.random.choice(consonants) + np.random.choice(vowels)
    return new_word
