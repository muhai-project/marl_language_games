import random

import numpy as np

from emrl.environment.lexicon import Lexicon
from emrl.utils.invention import make_id

SPEAKER = "SPEAKER"
HEARER = "HEARER"


class Agent:
    def __init__(self, cfg, world):
        self.cfg = cfg
        self.id = make_id("AG")
        self.lexicon = Lexicon(cfg)
        self.world = world

    def reset(self, context, topic):
        self.context = context
        self.topic = topic
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
            return max(actions, key=lambda sa_pair: sa_pair.q_val)  # todo
        else:
            return random.sample(actions, k=1)[0]

    def invention_strategy(self, meanings):
        """Strategy of invention by an agent.

        This following strategy is described in Chapter 5 in Lexicon Formation in Autonomous Robots by Loetzsch.
        Only when production completely fails, i.e. the speaker does not have a word for any
        of the conceptualized meanings, a new word is created for a randomly chosen meaning and production
        is retried again.
        """
        return random.sample(meanings, k=1)[0]

    def find_in_context(self, actions):
        """Returns a subset (action masking) of the given actions that is consistent with the current context."""
        context_actions = []
        for obj in self.context:
            categories = self.world.get_categories(obj)
            possible_objects = list(
                filter(lambda sa_pair: sa_pair.meaning in categories, actions)
            )
            if len(possible_objects) == 1:
                # corresponds to a path as Loetzsch describes [form -> category -> obj]
                choice = (
                    possible_objects[0],
                    obj,
                )
                context_actions.append(choice)
        return context_actions

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
            best_action = self.epsilon_greedy(actions, eps=self.cfg.LEARNING_RATE)
            return best_action.form
        else:
            return None

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

    def produce(self, meanings):
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
        actions = self.lexicon.get_actions_produce(meanings)
        if actions:
            # select action with highest q-value
            best_action = self.epsilon_greedy(actions, eps=self.cfg.LEARNING_RATE)
        else:
            # invent a new sa_pair for the meaning
            meaning = self.invention_strategy(meanings)
            best_action = self.lexicon.invent_sa_pair(meaning)
            invented = True
        self.applied_sa_pair = best_action
        return best_action.form, invented

    def comprehend(self, utterance):
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
        parsed_lexs = self.lexicon.get_actions_comprehend(utterance)
        actions = self.find_in_context(parsed_lexs)
        if actions:
            # selection action with highest q-value
            # note: given actions are (cxn - topic) tuples, hence path[0].q_val, TODO readability
            best_action = max(actions, key=lambda path: path[0].q_val)
            self.applied_sa_pair, self.topic = best_action
        return parsed_lexs

    def reconceptualize_and_adopt(self, topic, form):
        """Adopts all discriminating categories of the given topic with the given form.

        Args:
            topic (str): the topic of the episode
            form (str): an utterance by another agent
        """
        discr_cats = self.world.conceptualize(topic, self.context)
        for other_meaning in discr_cats:
            self.lexicon.adopt_sa_pair(other_meaning, form)

    def adopt(self, meaning, form):
        """Adopts the association of meaning and form to the lexicon of the agent."""
        if self.applied_sa_pair is None:
            self.reconceptualize_and_adopt(meaning, form)

    def update_q(self, sa_pair, reward):
        """Updates the q-value of the given state/action pair."""
        old_q = sa_pair.q_val
        # no discount as it is a bandit
        new_q = old_q + self.cfg.LEARNING_RATE * (reward - old_q)
        sa_pair.q_val = new_q
        if sa_pair.q_val < self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE:
            self.lexicon.remove_sa_pair(sa_pair)

    def lateral_inhibition(self, primary_cxn):
        sa_pairs = self.lexicon.get_actions_produce(primary_cxn.meaning)
        sa_pairs.remove(primary_cxn)
        for sa_pair in sa_pairs:
            self.update_q(sa_pair, self.cfg.REWARD_FAILURE)

    def align(self):
        """Align the q-table of the agent with the given reward if and only if
        an action was chosen (applied_sa_pair)."""
        if self.applied_sa_pair and self.communicative_success:
            self.update_q(self.applied_sa_pair, self.cfg.REWARD_SUCCESS)
            # self.lateral_inhibition(self.applied_sa_pair)
        elif self.applied_sa_pair:
            self.update_q(self.applied_sa_pair, self.cfg.REWARD_FAILURE)

    def __str__(self):
        return f"Agent id: {self.id}"
