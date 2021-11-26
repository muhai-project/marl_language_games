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

    def reset(self):
        self.communicative_success = True
        self.applied_sa_pair = None

    def epsilon_greedy(self, actions, eps):
        """Approach to balance exploitation vs exploration.

        If eps = 0, the approach is deterministic, i.e. no exploration.

        Args:
            actions (list): state/action pairs
            eps (float): epsilon value determines the chance of exploring

        Returns:
            sa_pair: a state/action pair from the given list of actions
        """
        p = np.random.random()
        if p < (1 - eps):
            return max(actions, key=lambda sa_pair: sa_pair.q_val)
        else:
            return np.random.choice(actions)

    def policy(self, role, state):
        """Find the best action to take given the current state and the role of the agent.

        This action selection mechanism is bidirectional and depends on the role of the
        agent during the linguistic interaction.

        Args:
            role (str): role of the agent - SPEAKER or HEARER
            state (str): state of the environment

        Returns:
            str: action response of the agent
        """
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)

    def produce(self, state):
        """Returns an utterance given the state of the environment.

        The state of the environment corresponds to a row in the q-table of the agent.
        If the agent does not have an entry for the given state, a new state/action pair is added.
        The action in this pair corresponds to a newly invented utterance.

        Args:
            state (str): the topic of the interaction

        Returns:
            str: an utterance
        """
        best_action = None
        invented = False  # logging purposes

        actions = self.lexicon.get_actions_produce(state)
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
        """Returns an interpretation of the state of the environment.

        The state of the environment corresponds to a column in the q-table of the agent.
        The interpretation then corresponds to the action with the greatest value.
        This interpretation is in the naming game the interpreted meaning of the agents' utterance.
        If no action can be found for the given

        Args:
            state (str): an utterance by another agent

        Returns:
            str: the interpreted meaning
        """
        actions = self.lexicon.get_actions_comprehend(state)
        if actions:
            # selection action with highest q_value
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY)
            self.applied_sa_pair = best_action
            return best_action.meaning
        else:
            # no actions found for the given state
            return None

    def adopt(self, topic, utterance):
        """Adopts the association of topic and utterance as a new state/aciton pair
        to the lexicon of the agent."""
        self.lexicon.adopt_sa_pair(topic, utterance)

    def update_q(self, sa_pair, reward):
        """Updates the q_value of the given state/action pair."""
        old_q = sa_pair.q_val
        # no discount as it is a bandit
        new_q = old_q + self.cfg.LEARNING_RATE * (reward - old_q)
        sa_pair.q_val = new_q
        if sa_pair.q_val < self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE:
            self.lexicon.remove_sa_pair(sa_pair)

    def lateral_inhibition(self):
        sa_pairs = self.lexicon.get_actions_produce(self.applied_sa_pair.meaning)
        sa_pairs.remove(self.applied_sa_pair)
        for sa_pair in sa_pairs:
            self.update_q(sa_pair, self.cfg.REWARD_FAILURE)

    def align(self):
        """Align the q-table of the agent with the given reward if and only if
        an action was chosen (applied_sa_pair)."""
        if self.applied_sa_pair:
            if self.communicative_success:
                self.update_q(self.applied_sa_pair, self.cfg.REWARD_SUCCESS)
                if self.cfg.LATERAL_INHIBITION:
                    self.lateral_inhibition()
            else:
                self.update_q(self.applied_sa_pair, self.cfg.REWARD_FAILURE)

    def print_lexicon(self):
        print(self.lexicon)

    def __str__(self):
        return f"Agent id: {self.id}"
