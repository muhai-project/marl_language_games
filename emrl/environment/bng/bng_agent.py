import numpy as np

from emrl.environment.lexicon import Lexicon
from emrl.utils.invention import make_id

SPEAKER = "SPEAKER"
HEARER = "HEARER"


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.id = make_id("AG")
        self.lexicon = Lexicon(self.cfg)
        self.communicative_success = True
        self.applied_sa_pair = None

    def epsilon_greedy(self, actions, eps):
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration."""
        # [RL] - has no conceptual bridge to LG, as there is no exploration in LG
        p = np.random.random()
        if p < (1 - eps):
            return max(actions, key=lambda sa_pair: sa_pair.q_val)
        else:
            return np.random.choice(actions)

    def policy(self, role, state):
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        # [RL] - language processing is now a policy which depends on the role of the agents
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)

    def produce(self, state):
        """Finds or invents a new state-action pair for the given meaning."""
        # state determines possible actions
        actions = self.lexicon.get_actions_produce(state)
        best_action = None
        invented = False
        if actions:
            # select action with highest q_value
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY)
        else:
            # invent a new sa_pair for the meaning
            best_action = self.lexicon.invent_sa_pair(state)
            invented = True
        self.applied_sa_pair = best_action
        return best_action.form, invented

    def comprehend(self, state):
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        # state determines possible actions
        actions = self.lexicon.get_actions_comprehend(state)
        if actions:
            # selection action with highest q_value
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY)
            self.applied_sa_pair = best_action
            return best_action.meaning
        return None

    def adopt(self, topic, utterance):
        """Adopts the association of topic and utterance to the lexicon of the agent."""
        self.lexicon.adopt_sa_pair(topic, utterance)

    def update_q(self, sa_pair, reward):
        """Updates the q_value of a state, action pair (a construction)."""
        old_q = sa_pair.q_val
        # no discount as it is a bandit
        new_q = old_q + self.cfg.LEARNING_RATE * (reward - old_q)
        sa_pair.q_val = new_q
        if sa_pair.q_val < self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE:
            self.lexicon.remove_sa_pair(sa_pair)

    def lateral_inhibition(self):
        # [LG] no conceptual/terminological bridge at the moment
        sa_pairs = self.lexicon.get_actions_produce(self.applied_sa_pair.meaning)
        sa_pairs.remove(self.applied_sa_pair)
        for sa_pair in sa_pairs:
            self.update_q(sa_pair, self.cfg.REWARD_FAILURE)

    def align(self):
        """Align the q-table of the agent with the given reward if and only if an action was chosen (applied_sa_pair)."""
        # [LG] - value iteration, mechanism by which optimal policy and values are computed
        if self.applied_sa_pair:
            if self.communicative_success:
                self.update_q(self.applied_sa_pair, self.cfg.REWARD_SUCCESS)
                self.lateral_inhibition()
            else:
                self.update_q(self.applied_sa_pair, self.cfg.REWARD_FAILURE)

    def print_lexicon(self):
        print(self.lexicon)

    def __str__(self):
        return f"Agent id: {self.id}"
