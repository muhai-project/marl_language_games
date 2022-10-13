import logging
import random
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable

from marl_language_games.environment.gg.gg_agent import HEARER, SPEAKER, Agent
from marl_language_games.utils.invention import make_id

SUCCESS = 0
FAIL_THEN_ADOPT = 1
FAIL_DUE_TO_DISCR = 2


class World:
    """Abstraction class of the part of the environment which handles the object-category mappings."""

    def __init__(self, world_size, amount_cats, cats_per_obj):
        """Initializes a world of objects represented by a set of categories."""
        self.cats = [make_id("CATEGORY") for i in range(amount_cats)]
        self.objs = defaultdict(list)
        for _ in range(world_size):
            self.objs[make_id("OBJECT")] = random.sample(self.cats, cats_per_obj)
        self.objects = list(self.objs.keys())

    def pick_topic(self, context):
        """Given a list of objects (context) returns at random one of the objects as the topic."""
        return random.sample(context, k=1)[0]

    def pick_context(self, context_min_size, context_max_size):
        """Given a world chooses a subset of the world of objects as the context.

        The size of the context is sampled uniformly at run-time using the given parameters context_min/max_size.
        """
        context_size = np.random.randint(context_min_size, context_max_size + 1)
        return random.sample(self.objects, k=context_size)

    def get_categories(self, obj):
        return self.objs[obj]

    def conceptualize(self, topic, context):
        """Returns the set of categories of the topic that is discriminating relative to the the context."""
        other_objects = [obj for obj in context if obj != topic]
        discr_cats = set(self.objs[topic])
        for obj in sorted(list(other_objects)):
            discr_cats -= set(self.objs[obj])
        return list(discr_cats)

    def get_contextualization_table(self, topic, context):
        tbl = PrettyTable()

        context.sort()
        context = sorted(context, key=len)
        sorted_context = {k: v for v, k in enumerate(context)}

        cats_dict = defaultdict(list)
        for obj in context:
            for cat in self.objs[obj]:
                cats_dict[cat].append(obj)

        rows = []
        for cat in self.cats:
            row = [""] * len(context)
            add_row = False
            for obj in context:
                if obj in cats_dict[cat]:
                    idx = sorted_context[obj]
                    row[idx] = "x"
                    add_row = True
            if add_row:
                row.insert(0, cat)
                rows.append(row)

        context.insert(0, "m/o")
        for idx, obj in enumerate(context, 0):
            if obj == topic:
                context[idx] = f"-> {obj} <-"
        tbl.field_names = context
        for row in rows:
            tbl.add_row(row)

        return str(tbl)


class GuessingGameEnv:
    """
    Guessing game environment

    The guessing game is a referential game where the goal is
    to achieve communicative success between a population of agents.
    In each episode (called a local interaction) of the game,
    two autonomous agents are selected from the population.
    The agents in the episode are assigned a role either of speaker or hearer.

    In this game, the environment (world) consists of n objects represented by a set of categories.
    In each episode, a subset of the world is selected as the context of the episode.
    One object of the context is then selected by the speaker as the topic.
    (In this environment, the topic is chosen randomly by sampling an object from the context.)
    The speaker's goal is to successfully communicate to the hearer the topic.
    In order to do so, the speaker internally needs to figure
    out which categories of the topic are discriminative
    in relation to set of objects of the context.

    The environment corresponds to the version of the 'guessing game' problem
    described in Chapter 5 of 'Lexicon Formation in Autonomous Robots' by Loetzsch.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.world = World(
            self.cfg.WORLD_SIZE,
            self.cfg.AMOUNT_CATEGORIES,
            self.cfg.CATEGORIES_PER_OBJECT,
        )
        self.context, self.topic, self.discriminative_attrs = None, None, None
        self.speaker, self.hearer = None, None
        self.population = [
            Agent(self.cfg, self.world) for i in range(self.cfg.POPULATION_SIZE)
        ]

    def reset(self, debug=False):
        """Resets the guessing game environment."""
        # choose interacting agents
        self.speaker, self.hearer = random.sample(self.population, k=2)

        # determine context, topic and the discriminating categories
        self.context = self.world.pick_context(
            self.cfg.CONTEXT_MIN_SIZE, self.cfg.CONTEXT_MAX_SIZE
        )
        self.topic = self.world.pick_topic(self.context)
        self.discriminative_attrs = self.world.conceptualize(self.topic, self.context)
        self.discriminative_attrs.sort()  # for visualization

        # reset agent
        self.speaker.reset(self.context, topic=self.topic)
        self.hearer.reset(self.context, topic=None)

        # logging
        self.lexicon_change = False
        self.lexicon_coherence = False

    def step(self, idx):
        """Interaction script of the guessing game

        Args:
            idx (int): denotes the ith interaction in the environment
        """
        outcome = None
        if self.discriminative_attrs:
            # speaker chooses action ifo topic
            utterance, self.lexicon_change = self.speaker.policy(
                SPEAKER, self.discriminative_attrs
            )
            # hearer chooses action ifo utterance
            interpretations = self.hearer.policy(HEARER, utterance)

            hearer_utterance = self.hearer.re_entrance_hearer(self.topic)  # monitoring
            self.lexicon_coherence = hearer_utterance == utterance  # monitoring

            # evaluate pulls
            if (
                interpretations is None
                or self.hearer.applied_sa_pair is None
                or (self.hearer.topic and self.hearer.topic != self.speaker.topic)
            ):
                self.lexicon_change = True
                self.hearer.adopt(self.speaker.topic, utterance)
                self.speaker.communicative_success = False
                self.hearer.communicative_success = False
                outcome = FAIL_THEN_ADOPT  # fail, adopt
            else:
                outcome = SUCCESS  # success
            # learn based on outcome
            self.speaker.align()
            self.hearer.align()

            self.print_example_interaction(idx, utterance, outcome)
        else:  # no discriminating categories for the topic
            self.speaker.communicative_success = False
            self.hearer.communicative_success = False
            outcome = FAIL_DUE_TO_DISCR

    def print_example_interaction(self, idx, utterance, outcome):
        logging.debug(f"\n\n- Episode {idx}")
        logging.debug(f"  ~~ GAME BETWEEN: {self.speaker.id} - {self.hearer.id} ~~")
        logging.debug(f"  ~~ CONTEXT: {sorted(self.context)} ~~")
        logging.debug(f"  ~~ TOPIC: {self.topic} ~~")
        logging.debug(self.world.get_contextualization_table(self.topic, self.context))
        logging.debug(
            f"  ~~ DISCRIMINATING CATEGORIES: {sorted(self.discriminative_attrs, key=len)} ~~"
        )
        logging.debug(f" === {self.speaker.id} q-table:")
        logging.debug(self.speaker.lexicon)
        logging.debug(f" === {self.speaker.id} uttered {utterance}")
        logging.debug(f" === {self.hearer.id} q-table:")
        logging.debug(self.hearer.lexicon)
        logging.debug(f" === {self.hearer.id} interpreted {self.hearer.topic}")

        if outcome == FAIL_THEN_ADOPT:
            logging.debug(" ===> FAILURE, hence adopting {utterance} <===")
        elif outcome == SUCCESS:
            logging.debug(" ===> SUCCESS <===")
        elif outcome == FAIL_DUE_TO_DISCR:
            logging.debug(" ===> FAILURE, due to no discrimination <===")
