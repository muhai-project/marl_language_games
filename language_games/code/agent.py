from collections import defaultdict
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
        self.applied_cxn = None
        self.learning_rate = exp.cfg.LEARNING_RATE
        self.eps_greedy = exp.cfg.EPS_GREEDY
        self.reward_success = exp.cfg.REWARD_SUCCESS
        self.reward_failure = exp.cfg.REWARD_FAILURE
        self.epsilon_failure = exp.cfg.EPSILON_FAILURE

    def epsilon_greedy(self, actions, eps):
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration."""
        p = np.random.random()
        if p < (1 - eps): 
            return max(actions, key=lambda cxn: cxn.q_val)
        else:
            return np.random.choice(actions)

    def policy(self, role, state):
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        if role == SPEAKER:
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
            self.applied_cxn = best_action
            return best_action.meaning
        return None
    
    def adopt(self, topic, utterance):
        """Adopts the association of topic and utterance to the lexicon of the agent."""
        self.lexicon.adopt_cxn(topic, utterance)
    
    def update_q(self, cxn, reward):
        """Updates the q_value of a state, action pair (a construction). """
        old_q = cxn.q_val
        new_q = old_q + self.learning_rate * (reward - old_q) # no discount as it is a bandit
        cxn.q_val = new_q
        if cxn.q_val < self.reward_failure + self.epsilon_failure:
            self.lexicon.remove_cxn(cxn)

    def lateral_inhibition(self):
        cxns = self.lexicon.get_cxns_with_meaning(self.applied_cxn.meaning)
        cxns.remove(self.applied_cxn)
        for cxn in cxns:
            self.update_q(cxn, self.reward_failure)
        
    def align(self):
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_cxn)."""
        if self.applied_cxn:
            if self.communicative_success:
                self.update_q(self.applied_cxn, self.reward_success)
                self.lateral_inhibition()
            else:
                self.update_q(self.applied_cxn, self.reward_failure)

    def print_lexicon(self):
        sorted_by_meaning = defaultdict(list)
        for item in sorted(self.lexicon.lexicon, key = lambda cxn: cxn.q_val):
            sorted_by_meaning[item.meaning].append(item)
        for key in sorted(sorted_by_meaning.keys()):
            for item in sorted_by_meaning[key]:
                print(item)

    def __str__(self):
        return f"Agent id: {self.id}"