from utils import invent

class Construction():
    def __init__(self, meaning, form):
        self.meaning = meaning
        self.form = form
        self.q_val = 0
        
    def __eq__(self, other):
        return self.meaning == other.meaning and self.form == other.form
    
    def __repr__(self):
        return f"Construction: ({self.meaning} - {self.form}) -> {self.q_val}"
        
class Lexicon():
    def __init__(self):
        self.lexicon = []
        
    def invent_cxn(self, meaning):
        new_cxn = Construction(meaning, invent())
        self.lexicon.append(new_cxn)
        return new_cxn
    
    def adopt_cxn(self, meaning, form):
        new_cxn = Construction(meaning, form)
        self.lexicon.append(new_cxn)
        return new_cxn
    
    def get_cxns_with_meaning(self, meaning):
        filtered = filter(lambda cxn: cxn.meaning == meaning, self.lexicon)
        return list(filtered)
    
    def get_cxns_with_form(self, form):
        filtered = filter(lambda cxn: cxn.form == form, self.lexicon)
        return list(filtered)

    def remove_cxn(self, cxn):
        self.lexicon.remove(cxn)

    def __len__(self):
        return len(self.lexicon)
    
    def __repr__(self):
        return f"{self.lexicon}"