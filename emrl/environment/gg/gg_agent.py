import numpy as np

from emrl.environment.lexicon import Lexicon
from emrl.utils.invention import make_id

SPEAKER = "SPEAKER"
HEARER = "HEARER"


class Agent:
    def __init__(self, cfg, world):
        self.id = make_id("AG")
        self.lexicon = Lexicon(cfg)
        self.communicative_success = True
        self.applied_sa_pair = None
        self.parsed_lexs = None
        self.context = None
        self.world = world
        self.learning_rate = cfg.LEARNING_RATE
        self.eps_greedy = cfg.EPS_GREEDY
        self.reward_success = cfg.REWARD_SUCCESS
        self.reward_failure = cfg.REWARD_FAILURE
        self.epsilon_failure = cfg.EPSILON_FAILURE

    def epsilon_greedy(self, actions, eps):
        """Approach to balance exploitation vs exploration. If eps = 0, there is no exploration.
        # [RL] - has no conceptual bridge to LG, as there is no exploration in LG"""
        p = np.random.random()
        if p < (1 - eps):
            return max(actions, key=lambda sa_pair: sa_pair.q_val)  # todo
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

    def policy(self, role, state):
        """The given state corresponds to a meaning or form depending on the role of the agent."""
        # [RL] - language processing is now a policy which depends on the role of the agent
        if role == SPEAKER:
            return self.produce(state)
        else:
            return self.comprehend(state)

    def produce(self, meanings):  # [LG] - action selection in one direction
        """Finds or invents an action for the given meaning."""
        # state determines possible actions
        actions = self.lexicon.get_actions_produce(meanings)
        best_action = None
        invented = False
        if actions:
            # [RL] - select action with highest q_value
            best_action = self.epsilon_greedy(actions, eps=self.eps_greedy)
        else:
            invented = True
            meaning = self.invention_strategy(meanings)
            # [LG] - if no entry (action) is found for the given states in the
            # current q-table, then create a new entry for one of the states.
            best_action = self.lexicon.invent_sa_pair(meaning)
            # [LG] - add a new entry in the q-table
        self.applied_sa_pair = best_action
        return best_action.form, invented

    def comprehend(self, utterance):  # [LG] - action selection in other direction
        """Interprets the action of a speaker (an utterance) and chooses a corresponding action."""
        # parse - state determines possible actions
        self.parsed_lexs = self.lexicon.get_actions_comprehend(utterance)
        # interpret - [LG] - action masking based on the the context
        actions = self.find_in_context(self.parsed_lexs)
        # [RL] - ACTION SELECTION - maximize q-val of the remanining unmasked actions
        if actions:
            # note: given actions are (cxn - topic) tuples, hence path[0].q_val, TODO readability
            best_action = max(actions, key=lambda path: path[0].q_val)
            self.applied_sa_pair = best_action[0]
            self.topic = best_action[1]
        return self.parsed_lexs

    def reconceptualize_and_adopt(self, topic, form):
        discr_cats = self.world.conceptualize(topic, self.context)
        for other_meaning in discr_cats:
            self.lexicon.adopt_sa_pair(other_meaning, form)

    def adopt(self, meaning, form):
        """Adopts the association of meaning and form to the lexicon of the agent."""
        # [LG] - adding a new state/action to the state/action space
        if self.applied_sa_pair is None:
            self.reconceptualize_and_adopt(meaning, form)

    def update_q(self, sa_pair, reward):  # [RL] update score - based on feedback
        """Updates the q_value of a state, action pair (a construction)."""
        old_q = sa_pair.q_val
        # no discount as it is a bandit
        new_q = old_q + self.learning_rate * (reward - old_q)
        sa_pair.q_val = new_q
        if sa_pair.q_val < self.reward_failure + self.epsilon_failure:
            self.lexicon.remove_sa_pair(sa_pair)

    def lateral_inhibition(self, primary_cxn):
        # [LG] no conceptual/terminological bridge at the moment
        sa_pairs = self.lexicon.get_actions_produce(primary_cxn.meaning)
        sa_pairs.remove(primary_cxn)
        for sa_pair in sa_pairs:
            self.update_q(sa_pair, self.reward_failure)

    def align(self):
        """Align the q-table of the agent with the given reward if and only
        if an action was chosen (applied_sa_pair)."""
        if self.applied_sa_pair and self.communicative_success:
            self.update_q(self.applied_sa_pair, self.reward_success)
            self.lateral_inhibition(self.applied_sa_pair)
        elif self.applied_sa_pair:
            self.update_q(self.applied_sa_pair, self.reward_failure)

    def print_lexicon(self):
        print(self.lexicon)

    def __str__(self):
        return f"Agent id: {self.id}"
