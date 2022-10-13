from collections import defaultdict

from prettytable import PrettyTable

from marl_language_games.utils.invention import invent


class SAPair:
    def __init__(self, meaning, form, initial_value=0):
        self.meaning = meaning
        self.form = form
        self.q_value = initial_value

    def __hash__(self):
        return hash((self.meaning, self.form))

    def __eq__(self, other):
        return self.meaning == other.meaning and self.form == other.form

    def __repr__(self):
        return f"SAPair: ({self.meaning} - {self.form}) -> {self.q_value}"


class Lexicon:
    """The bidirectional dynamic Q-table implemented as a list of state-action pairs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.q_table = []  # the set of state/action pairs, i.e. the q-table

    def invent_sa_pair(self, state):
        """Invents an action for a given state and adds the new pair to the lexicon.

        Args:
            meaning (str): denotes the meaning of an object

        Returns:
            sa_pair: the newly added state/action pair of the lexicon
        """
        new_sa_pair = SAPair(state, invent(), self.cfg.INITIAL_Q_VALUE)
        self.q_table.append(new_sa_pair)
        return new_sa_pair

    def adopt_sa_pair(self, meaning, form):
        """Adds a given state/action pair to the lexicon.

        The value of the new pair is initialized using the config.

        Args:
            meaning (str): denotes the meaning of an object
            form (str): denotes the utterance to describe the object

        Returns:
            sa_pair: the newly added state/action pair of the lexicon
        """
        new_sa_pair = SAPair(meaning, form, self.cfg.INITIAL_Q_VALUE)
        # uses SAPair __eq__ to determine if member
        if new_sa_pair not in self.q_table:
            self.q_table.append(new_sa_pair)
        return new_sa_pair

    def get_actions_produce(self, states):
        """Returns the set of possible state/action pairs that have the given state.

        The state in this case corresponds to a meaning of an object.

        Args:
            states (str or list): a single meaning or a list of meanings

        Returns:
            list: a list of all state/action pairs that are a match
        """
        filtered = filter(lambda sa_pair: sa_pair.meaning in states, self.q_table)
        return list(filtered)

    def get_actions_comprehend(self, states):
        """Returns the set of possible state/action pairs that have the given state.

        The state in this case corresponds to the form used to describe a meaning.

        Args:
            states (str or list): a single form or a list of form

        Returns:
            list: a list of all state/action pairs that are a match
        """
        filtered = filter(lambda sa_pair: sa_pair.form in states, self.q_table)
        return list(filtered)

    def remove_sa_pair(self, sa_pair):
        """Removes a state/action pair from the lexicon."""
        self.q_table.remove(sa_pair)

    def __len__(self):
        """Returns the length of the q-table, which corresponds to the amount of current entries."""
        return len(self.q_table)

    def __repr__(self):
        """Returns a string representation of the lexicon as a bidirectional dynamic q-table."""
        tbl = PrettyTable()

        forms = sorted(list(set([cxn.form for cxn in self.q_table])))
        forms = {k: v for v, k in enumerate(forms)}

        meanings = defaultdict(list)
        for cxn in self.q_table:
            meanings[cxn.meaning].append(cxn)

        rows = []
        meaning_keys = list(meanings.keys())
        meaning_keys.sort()
        meaning_keys = sorted(meaning_keys, key=len)
        for meaning in meaning_keys:
            cxns = meanings[meaning]
            row = [""] * len(forms)
            for cxn in cxns:
                idx = forms[cxn.form]
                row[idx] = round(cxn.q_value, 3)
            row.insert(0, meaning)
            rows.append(row)

        forms = list(forms.keys())
        forms.insert(0, "m/f")

        tbl.field_names = forms
        for row in rows:
            tbl.add_row(row)

        return str(tbl)
