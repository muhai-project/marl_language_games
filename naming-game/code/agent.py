import numpy as np

from utils import make_id
from lexicon import Lexicon

SPEAKER = "SPEAKER"
HEARER = "HEARER"

class Agent():
    def __init__(self, exp):
        self.id = make_id("AG")
        self.lexicon = Lexicon()
        self.communicative_success = True
        self.role = None
        self.applied_cxn = None
        self.learning_rate = exp.cfg.LEARNING_RATE
        self.eps_greedy = exp.cfg.EPS_GREEDY 

    def is_speaker(self):
        return self.role == SPEAKER

    def epsilon_greedy(self, actions, eps):
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration."""
        p = np.random.random()
        if p < (1 - eps): 
            return max(actions, key=lambda cxn: cxn.q_val)
        else:
            return np.random.choice(actions)

    def policy(self, state):
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        if self.is_speaker():
            return self.produce(state)
        else:
            return self.comprehend(state)
    
    def produce(self, meaning):
        """Finds or invents an action (a cxn) for the given meaning."""
        actions = self.lexicon.get_cxns_with_meaning(meaning) # state determines possible actions
        best_action = None
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.eps_greedy) # select action with highest q_value
        else:
            best_action = self.lexicon.invent_cxn(meaning) # invent a new cxn for the meaning
        self.applied_cxn = best_action
        return best_action.form
    
    def comprehend(self, utterance):
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        actions = self.lexicon.get_cxns_with_form(utterance) # state determines possible actions
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.eps_greedy) # selection action with highest q_value
            return best_action.meaning
        return None
    
    def adopt(self, topic, utterance):
        """Adopts the association of topic and utterance to the lexicon of the agent."""
        self.lexicon.adopt_cxn(topic, utterance)
    
    def align(self, reward):
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_cxn)."""
        if self.applied_cxn:
            old_q = self.applied_cxn.q_val
            new_q = old_q + self.learning_rate * (reward - old_q) # no discount as it is a bandit
            self.applied_cxn.q_val = new_q

    def __str__(self):
        return f"Agent id: {self.id} - role: {self.role} - lexicon: {self.lexicon}"