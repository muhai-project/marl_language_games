from collections import defaultdict
import numpy as np

from utils import make_id
from lexicon import Lexicon

SPEAKER = "SPEAKER"
HEARER = "HEARER"

class Agent():
    def __init__(self, cfg):
        self.cfg = cfg
        self.id = make_id("AG")
        self.lexicon = Lexicon()
        self.communicative_success = True
        self.applied_cxn = None

    def epsilon_greedy(self, actions, eps): # [RL] - has no conceptual bridge to LG, as there is no exploration in LG
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration."""
        p = np.random.random()
        if p < (1 - eps): 
            return max(actions, key=lambda cxn: cxn.q_val)
        else:
            return np.random.choice(actions)

    def policy(self, role, state): # [RL] - language processing is now a policy which depends on the role of the agent
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)
    
    def produce(self, meaning): # [LG] - action selection in one direction
        """Finds or invents an action (a cxn) for the given meaning."""
        actions = self.lexicon.get_cxns_with_meaning(meaning) # state determines possible actions
        best_action = None
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY) # select action with highest q_value
        else:
            best_action = self.lexicon.invent_cxn(meaning) # invent a new cxn for the meaning
        self.applied_cxn = best_action
        return best_action.form
    
    def comprehend(self, utterance): # [LG] - action selection in other direction
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        actions = self.lexicon.get_cxns_with_form(utterance) # state determines possible actions
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY) # selection action with highest q_value
            self.applied_cxn = best_action
            return best_action.meaning
        return None
    
    def adopt(self, topic, utterance): # [LG] - adding a new state/action to the state/action space
        """Adopts the association of topic and utterance to the lexicon of the agent."""
        self.lexicon.adopt_cxn(topic, utterance)

    def update_q(self, cxn, reward): # [RL] update score - based on feedback
        """Updates the q_value of a state, action pair (a construction). """
        old_q = cxn.q_val
        new_q = old_q + self.cfg.LEARNING_RATE * (reward - old_q) # no discount as it is a bandit
        cxn.q_val = new_q
        if cxn.q_val < self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE:
            self.lexicon.remove_cxn(cxn)

    def lateral_inhibition(self): # [LG] no conceptual/terminological bridge at the moment
        cxns = self.lexicon.get_cxns_with_meaning(self.applied_cxn.meaning)
        cxns.remove(self.applied_cxn)
        for cxn in cxns:
            self.update_q(cxn, self.cfg.REWARD_FAILURE)
        
    def align(self): # [LG] - value iteration, mechanism by which optimal policy and values are computed
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_cxn)."""
        if self.applied_cxn: # required to deal with not punishing adoption immediately
            if self.communicative_success:
                self.update_q(self.applied_cxn, self.cfg.REWARD_SUCCESS)
                self.lateral_inhibition()
            else:
                self.update_q(self.applied_cxn, self.cfg.REWARD_FAILURE)

    def print_lexicon(self):
        sorted_by_meaning = defaultdict(list)
        for item in sorted(self.lexicon.lexicon, key = lambda cxn: cxn.q_val):
            sorted_by_meaning[item.meaning].append(item)
        for key in sorted(sorted_by_meaning.keys()):
            for item in sorted_by_meaning[key]:
                print(item)

    def __str__(self):
        return f"Agent id: {self.id}"