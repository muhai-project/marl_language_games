from emrl.utils.invention import invent


class Construction:
    def __init__(self, meaning, form, initial_value=0):
        self.meaning = meaning  # [LG] - state/action
        self.form = form  # [LG] - state/action
        self.q_val = initial_value  # [RL] - value initialisation

    def __eq__(self, other):
        return self.meaning == other.meaning and self.form == other.form

    def __repr__(self):
        return f"Construction: ({self.meaning} - {self.form}) -> {self.q_val}"


class Lexicon:
    """Naive implementation of a lexicon as a list of constructions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.q_table = []  # [LG] - the set of state/action pairs, i.e. the q-table

    def invent_cxn(self, meaning):  # [LG] - adding a state/action pair
        new_cxn = Construction(meaning, invent(), self.cfg.INITIAL_Q_VAL)
        self.q_table.append(new_cxn)
        return new_cxn

    def adopt_cxn(self, meaning, form):  # [LG] - adding a state/action pair
        new_cxn = Construction(meaning, form, self.cfg.INITIAL_Q_VAL)
        # uses Construction __eq__ to determine if member
        if new_cxn not in self.q_table:
            self.q_table.append(new_cxn)
        return new_cxn

    def get_cxns_with_meaning(self, meanings):
        # [LG] - retrieving all actions given a state
        filtered = filter(lambda cxn: cxn.meaning in meanings, self.q_table)
        return list(filtered)

    def get_cxns_with_form(self, form):  # [LG] - retrieving all actions given a state
        filtered = filter(lambda cxn: cxn.form == form, self.q_table)
        return list(filtered)

    def remove_cxn(self, cxn):  # [LG] - removing a state/action pair
        self.q_table.remove(cxn)

    def __len__(self):
        return len(self.q_table)

    def __repr__(self):
        table = ""
        return table
        # return f"{self.q_table}"
