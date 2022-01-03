import random

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

    def reset(self, context):
        self.communicative_success = True
        self.applied_sa_pair = None
        self.context = context

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
            return random.sample(actions, k=1)[0]

    def find_in_context(self, actions):
        """Returns a subset (action masking) of the given actions that is consistent with the current context."""
        context_actions = filter(lambda action: action.meaning in self.context, actions)
        return list(context_actions)

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
        invented = False  # logging purposes
        best_action = None
        actions = self.lexicon.get_actions_produce(state)
        if actions:
            # select action with highest q-value
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY)
        else:
            # invent a new sa_pair for the meaning
            best_action = self.lexicon.invent_sa_pair(state)
            invented = True
        self.applied_sa_pair = best_action
        return best_action.form, invented

    def produce_as_hearer(self, state):
        """Returns an utterance given the state of the environment as hearer.

        The state of the environment corresponds to a row in the q-table of the agent.

        Serves to monitor lexicon coherence between interacting agents.

        Args:
            state (str): the topic of the interaction

        Returns:
            str or None: an utterance or none if the hearer could not produce for the state
        """
        actions = self.lexicon.get_actions_produce(state)
        actions = self.find_in_context(actions)
        if actions:
            best_action = self.epsilon_greedy(actions, eps=self.cfg.EPS_GREEDY)
            return best_action.form
        else:
            return None

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
            # selection action with highest q-value
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

    def update(self, sa_pair, reward):
        """Updates the Q-value of a given state-action pair using the specified update rule.

        Note: make sure cfg.UPDATE_RULE also matches up with the specific init values for:
            cfg.INITIAL_Q_VAL
            cfg.REWARD_SUCCESS
            cfg.REWARD_FAILURE

        For example, for the basic strategy:
            cfg.UPDATE_RULE: basic
            cfg.INITIAL_Q_VAL: 0.5
            cfg.REWARD_SUCCESS: 0.1 (~ delta_inc = 0.1)
            cfg.REWARD_FAILURE: -0.1 (~ delta_dec = 0.1)

        Args:
            sa_pair: the newly added state/action pair of the lexicon
            reward (float): reward associated with the update

        Raises:
            ValueError: True, if the given update rule is not implemented
        """
        if self.cfg.UPDATE_RULE == "interpolated":
            self.update_q(sa_pair, reward)
        elif self.cfg.UPDATE_RULE == "basic":
            self.update_basic(sa_pair, reward)
        else:
            raise ValueError(f"Given update rule {self.cfg.UPDATE_RULE} is not valid!")

    def remove_sa_pair(self, sa_pair):
        """Removes state-action pair from lexicon if and only if it is allowed.

        Args:
            sa_pair (SA_Pair): the newly added state/action pair of the lexicon
        """
        if self.cfg.DELETE_SA_PAIR:
            self.lexicon.remove_sa_pair(sa_pair)

    def update_basic(self, sa_pair, delta):
        """Updates the q-value of the given state/action pair using the basic update rule."""
        old_q = sa_pair.q_val
        new_q = old_q + delta
        if new_q >= 1:
            new_q = 1
        elif new_q <= 0:
            new_q = 0

        sa_pair.q_val = new_q
        if sa_pair.q_val <= 0:
            self.remove_sa_pair(sa_pair)

    def update_q(self, sa_pair, reward):
        """Updates the q-value of the given state/action pair using the interpolated update rule."""
        old_q = sa_pair.q_val
        new_q = old_q + self.cfg.LEARNING_RATE * (reward - old_q)
        sa_pair.q_val = new_q
        if sa_pair.q_val < self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE:
            self.remove_sa_pair(sa_pair)

    def lateral_inhibition(self):
        sa_pairs = self.lexicon.get_actions_produce(self.applied_sa_pair.meaning)
        sa_pairs.remove(self.applied_sa_pair)
        for sa_pair in sa_pairs:
            self.update(sa_pair, self.cfg.REWARD_FAILURE)

    def align(self):
        """Align the q-table of the agent with the given reward if and only if
        an action was chosen (applied_sa_pair)."""
        if self.applied_sa_pair:
            if self.communicative_success:
                self.update(self.applied_sa_pair, self.cfg.REWARD_SUCCESS)
                if self.cfg.LATERAL_INHIBITION:
                    self.lateral_inhibition()
            else:
                self.update(self.applied_sa_pair, self.cfg.REWARD_FAILURE)

    def __str__(self):
        return f"Agent id: {self.id}"
