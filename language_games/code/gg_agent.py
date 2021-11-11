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

    def epsilon_greedy(self, actions, eps):
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

    def policy(self, role, state):
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)
    
    def produce(self, meanings):
        """Finds or invents an action (a cxn) for the given meaning."""
        actions = self.lexicon.get_cxns_with_meaning(meanings) # state determines possible actions
        best_action = None
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.eps_greedy) # select action with highest q_value
        else:
            meaning = self.invention_strategy(meanings)
            best_action = self.lexicon.invent_cxn(meaning) # invent a new cxn for the meaning
        self.applied_cxn = best_action
        return best_action.form

    def comprehend(self, utterance):
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        # parse
        self.parsed_lexs = self.lexicon.get_cxns_with_form(utterance) # state determines possible actions
        # interpret
        actions = self.find_in_context(self.parsed_lexs)
        if actions:
            best_action = max(actions, key=lambda cxn: cxn[0].q_val)
            self.applied_cxn = best_action[0]
            self.topic = best_action[1]
        return self.parsed_lexs

    def find_in_context(self, actions):
        context_actions = []
        for obj in self.context:
            categories = self.world.get_categories(obj)
            possible_objects = list(filter(lambda cxn: cxn.meaning in categories, actions))
            if len(possible_objects) == 1:
                choice = (possible_objects[0], obj)
                context_actions.append(choice)
        return context_actions

    def other_paths(self, topic):
        for obj in self.context:
            categories = self.world.get_categories(obj)
            possible_objects = list(filter(lambda cxn: cxn.meaning in categories, self.parsed_lexs))
            if len(possible_objects) == 1 and obj == topic:
                return possible_objects[0]
        return False
        
    def reconceptualize_and_adopt(self, meaning, form):
        discr_cats = self.world.conceptualize(meaning, self.context)
        for other_meaning in discr_cats:
            self.lexicon.adopt_cxn(other_meaning, form)

    def adopt(self, meaning, form):
        """Adopts the association of meaning and form to the lexicon of the agent."""
        other_path = self.other_paths(meaning)
        if self.applied_cxn is None or not other_path:
            self.reconceptualize_and_adopt(meaning, form)
        elif other_path: # there was an another path
            self.correct_path = other_path

    def update_q(self, cxn, reward):
        """Updates the q_value of a state, action pair (a construction). """
        old_q = cxn.q_val
        new_q = old_q + self.learning_rate * (reward - old_q) # no discount as it is a bandit
        cxn.q_val = new_q
        # new_q = old_q + reward
        # cxn.q_val = max(min(100, new_q), 0)
        # if cxn.q_val <= 0 + self.epsilon_failure:
        if cxn.q_val < self.reward_failure + self.epsilon_failure:
            self.lexicon.remove_cxn(cxn)

    def lateral_inhibition(self, primary_cxn):
        cxns = self.lexicon.get_cxns_with_meaning(primary_cxn.meaning)
        cxns.remove(primary_cxn)
        for cxn in cxns:
            self.update_q(cxn, self.reward_failure)
        
    def align(self):
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_cxn).
        
        applied_cxn: check if the action was not added in the interaction (invention/adoption).
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