from utils import invent

class Construction():
    def __init__(self, meaning, form):
        self.meaning = meaning # [LG] - state/action
        self.form = form # [LG] - state/action
        self.q_val = 0 # [RL] - value initialisation
        
    def __eq__(self, other):
        return self.meaning == other.meaning and self.form == other.form
    
    def __repr__(self):
        return f"Construction: ({self.meaning} - {self.form}) -> {self.q_val}"
        
class Lexicon():
    """Naive implementation of a lexicon as a list of constructions."""
    def __init__(self):
        self.lexicon = [] # [LG] - the set of state/action pairs, i.e. the q-table
        
    def invent_cxn(self, meaning): # [LG] - adding a state/action pair
        new_cxn = Construction(meaning, invent()) 
        self.lexicon.append(new_cxn)
        return new_cxn
    
    def adopt_cxn(self, meaning, form): # [LG] - adding a state/action pair
        new_cxn = Construction(meaning, form)
        if new_cxn not in self.lexicon: # uses Construction __eq__ to determine if member 
            self.lexicon.append(new_cxn)
        return new_cxn
    
    def get_cxns_with_meaning(self, meanings): # [LG] - retrieving all actions given a state
        filtered = filter(lambda cxn: cxn.meaning in meanings, self.lexicon)
        return list(filtered)
    
    def get_cxns_with_form(self, form): # [LG] - retrieving all actions given a state
        filtered = filter(lambda cxn: cxn.form == form, self.lexicon)
        return list(filtered)

    def remove_cxn(self, cxn): # [LG] - removing a state/action pair
        self.lexicon.remove(cxn)

    def __len__(self):
        return len(self.lexicon)
    
    def __repr__(self):
        return f"{self.lexicon}"