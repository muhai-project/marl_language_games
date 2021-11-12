from collections import defaultdict
import numpy as np

from utils import make_id
from lexicon import Lexicon

SPEAKER = "SPEAKER"
HEARER = "HEARER"

class Agent():
    def __init__(self, cfg, world):
        self.id = make_id("AG")
        self.lexicon = Lexicon()
        self.communicative_success = True
        self.applied_cxn = None
        self.parsed_lexs = None
        self.correct_path = None
        self.context = None
        self.world = world
        self.learning_rate = cfg.LEARNING_RATE
        self.eps_greedy = cfg.EPS_GREEDY
        self.reward_success = cfg.REWARD_SUCCESS
        self.reward_failure = cfg.REWARD_FAILURE
        self.epsilon_failure = cfg.EPSILON_FAILURE

    def epsilon_greedy(self, actions, eps): # [RL] - has no conceptual bridge to LG, as there is no exploration in LG
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration."""
        p = np.random.random()
        if p < (1 - eps): 
            return max(actions, key=lambda cxn: cxn.q_val) # todo
        else:
            return np.random.choice(actions)

    def invention_strategy(self, meanings):
        """Strategy of invention by an agent.
        
        This following strategy is described in Chapter 5 in Lexicon Formation in Autonomous Robots by Loetzsch.
        Only when production completely fails, i.e. the speaker does not have a word for any
        of the conceptualized meanings, a new word is created for a randomly chosen meaning and production
        is retried again.
        """
        return np.random.choice(meanings)

    def find_in_context(self, actions):
        """Returns a subset of the given actions that is consistent with the current context."""
        context_actions = [] # [LG] - (~ action mask) all actions that are consistent with the current context
        for obj in self.context:
            categories = self.world.get_categories(obj)
            possible_objects = list(filter(lambda cxn: cxn.meaning in categories, actions))
            if len(possible_objects) == 1:
                choice = (possible_objects[0], obj) # corresponds to a path as Loetzsch describes [form -> category -> obj]
                context_actions.append(choice)
        return context_actions

    def policy(self, role, state): # [RL] - language processing is now a policy which depends on the role of the agent
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)
    
    def produce(self, meanings): # [LG] - action selection in one direction
        """Finds or invents an action (a cxn) for the given meaning."""
        actions = self.lexicon.get_cxns_with_meaning(meanings) # state determines possible actions
        best_action = None
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.eps_greedy) # [RL] - select action with highest q_value
        else:
            meaning = self.invention_strategy(meanings) # [LG] - if no entry (action) is found for the given states in the current q-table, then create a new entry for one of the states. 
            best_action = self.lexicon.invent_cxn(meaning) # [LG] - add a new entry in the q-table
        self.applied_cxn = best_action
        return best_action.form

    def comprehend(self, utterance): # [LG] - action selection in other direction
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        # parse
        self.parsed_lexs = self.lexicon.get_cxns_with_form(utterance) # state determines possible actions
        # interpret
        actions = self.find_in_context(self.parsed_lexs) # [LG] - action masking based on the the context 
        if actions: # [RL] - ACTION SELECTION - maximize q-val of the remanining unmasked actions
            best_action = max(actions, key=lambda path: path[0].q_val) # note: given actions are (cxn - topic) tuples, hence path[0].q_val, TODO readability
            self.applied_cxn = best_action[0]
            self.topic = best_action[1]
        return self.parsed_lexs

    def other_paths(self, topic):
        """Find an alternative action that leads to the given topic"""
        for obj in self.context:
            categories = self.world.get_categories(obj)
            possible_objects = list(filter(lambda cxn: cxn.meaning in categories, self.parsed_lexs))
            if len(possible_objects) == 1 and obj == topic:
                return possible_objects[0]
        return False
        
    def reconceptualize_and_adopt(self, topic, form):
        discr_cats = self.world.conceptualize(topic, self.context)
        for other_meaning in discr_cats:
            self.lexicon.adopt_cxn(other_meaning, form) # will not introduce duplicates

    def adopt(self, meaning, form):  # [LG] - adding a new state/action to the state/action space
        """Adopts the association of meaning and form to the lexicon of the agent."""
        other_path = self.other_paths(meaning)
        if self.applied_cxn is None or not other_path:
            self.reconceptualize_and_adopt(meaning, form)
        elif other_path: # there was an another path
            self.correct_path = other_path

    def update_q(self, cxn, reward):  # [RL] update score - based on feedback
        """Updates the q_value of a state, action pair (a construction)."""
        old_q = cxn.q_val
        new_q = old_q + self.learning_rate * (reward - old_q) # no discount as it is a bandit
        cxn.q_val = new_q
        # new_q = old_q + reward
        # cxn.q_val = max(min(100, new_q), 0)
        # if cxn.q_val <= 0 + self.epsilon_failure:
        if cxn.q_val < self.reward_failure + self.epsilon_failure:
            self.lexicon.remove_cxn(cxn)

    def lateral_inhibition(self, primary_cxn): # [LG] no conceptual/terminological bridge at the moment
        cxns = self.lexicon.get_cxns_with_meaning(primary_cxn.meaning)
        cxns.remove(primary_cxn)
        for cxn in cxns:
            self.update_q(cxn, self.reward_failure)
        
    def align(self):
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_cxn).
        
        applied_cxn: 
            check if the action was not added during the interaction (invention/adoption).
        correct_path: another path to the topic is found
            If true, this action (cxn) is treated specially in consolidation, i.e. treated as if it was succesful.
        """
        if self.applied_cxn and self.communicative_success:
            self.update_q(self.applied_cxn, self.reward_success)
            self.lateral_inhibition(self.applied_cxn)
        else:
            if self.correct_path:
                self.update_q(self.correct_path, self.reward_success)
                self.lateral_inhibition(self.correct_path)
            elif self.applied_cxn:
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