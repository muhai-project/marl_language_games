from collections import defaultdict

from prettytable import PrettyTable

from emrl.utils.invention import invent


class Construction:
    def __init__(self, meaning, form, initial_value=0):
        self.meaning = meaning  # [LG] - state/action
        self.form = form  # [LG] - state/action
        self.q_val = initial_value  # [RL] - value initialisation

    def __hash__(self):
        return hash((self.meaning, self.form))

    def __eq__(self, other):
        return self.meaning == other.meaning and self.form == other.form

    def __repr__(self):
        return f"Construction: ({self.meaning} - {self.form}) -> {self.q_val}"


class Lexicon:
    """Naive implementation of a lexicon as a list of constructions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.q_table = []  # [LG] - the set of state/action pairs, i.e. the q-table

    def invent_sa_pair(self, meaning):  # [LG] - adding a state/action pair
        new_sa_pair = Construction(meaning, invent(), self.cfg.INITIAL_Q_VAL)
        self.q_table.append(new_sa_pair)
        return new_sa_pair

    def adopt_sa_pair(self, meaning, form):  # [LG] - adding a state/action pair
        new_sa_pair = Construction(meaning, form, self.cfg.INITIAL_Q_VAL)
        # uses Construction __eq__ to determine if member
        if new_sa_pair not in self.q_table:
            self.q_table.append(new_sa_pair)
        return new_sa_pair

    def get_actions_produce(self, meanings):
        # [LG] - retrieving all actions given a state
        filtered = filter(lambda sa_pair: sa_pair.meaning in meanings, self.q_table)
        return list(filtered)

    def get_actions_comprehend(self, form):
        filtered = filter(lambda sa_pair: sa_pair.form == form, self.q_table)
        return list(filtered)

    def remove_sa_pair(self, sa_pair):
        self.q_table.remove(sa_pair)

    def __len__(self):
        return len(self.q_table)

    def __repr__(self):
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
                row[idx] = round(cxn.q_val, 3)
            row.insert(0, meaning)
            rows.append(row)

        forms = list(forms.keys())
        forms.insert(0, "m/f")

        tbl.field_names = forms
        for row in rows:
            tbl.add_row(row)

        return str(tbl)
