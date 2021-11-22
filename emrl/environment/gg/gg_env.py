import random
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable

from emrl.environment.environment import Environment
from emrl.environment.gg.gg_agent import HEARER, SPEAKER, Agent
from emrl.utils.invention import make_id


class World:
    """Abstraction class of the part of the environment which handles the object-category mappings."""

    def __init__(self, world_size, amount_cats, cats_per_obj):
        """Initializes a world of objects represented by a set of categories."""
        self.cats = [make_id("C") for i in range(amount_cats)]
        self.objs = defaultdict(list)
        for _ in range(world_size):
            self.objs[make_id("OBJ")] = random.sample(self.cats, cats_per_obj)
        self.objects = list(self.objs.keys())

    def pick_topic(self, context):
        """Given a list of objects (context) returns at random one of the objects as the topic."""
        return np.random.choice(context)

    def pick_context(self, context_min_size, context_max_size):
        """Given a world chooses a subset of the world of objects as the context.

        The size of the context is sampled uniformly at run-time using the given parameters context_min/max_size.
        """
        context_size = np.random.randint(context_min_size, context_max_size + 1)
        return np.random.choice(self.objects, size=context_size, replace=False)

    def get_categories(self, obj):
        return self.objs[obj]

    def conceptualize(self, topic, context):
        """Returns the set of categories of the topic that is discriminating relative to the the context."""
        other_objects = [obj for obj in context if obj != topic]
        discr_cats = set(self.objs[topic])
        for obj in sorted(list(other_objects)):
            discr_cats -= set(self.objs[obj])
        return list(discr_cats)

    def print_contextualization(self, topic, context):
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

        print(tbl)


class GuessingGameEnv(Environment):
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
        super().__init__()
        self.cfg = cfg
        self.world = World(
            self.cfg.WORLD_SIZE,
            self.cfg.AMOUNT_CATEGORIES,
            self.cfg.CATEGORIES_PER_OBJECT,
        )
        self.context, self.topic, self.discriminative_cats = None, None, None
        self.speaker, self.hearer = None, None
        self.population = [
            Agent(self.cfg, self.world) for i in range(self.cfg.POPULATION_SIZE)
        ]

    def reset(self, debug=False):
        """Resets the guessing game environment."""
        # choose interacting agents
        self.speaker, self.hearer = np.random.choice(
            self.population, size=2, replace=False
        )

        self.context = self.world.pick_context(
            self.cfg.CONTEXT_MIN_SIZE, self.cfg.CONTEXT_MAX_SIZE
        )
        self.topic = self.world.pick_topic(self.context)
        self.discriminative_cats = self.world.conceptualize(self.topic, self.context)
        self.discriminative_cats.sort()  # for visualization

        # reset agent
        self.speaker.context, self.hearer.context = self.context, self.context
        self.speaker.applied_cxn, self.hearer.applied_cxn = None, None
        self.speaker.parsed_lexs, self.hearer.parsed_lexs = None, None
        self.speaker.topic, self.hearer.topic = self.topic, None
        self.speaker.communicative_success, self.hearer.communicative_success = (
            True,
            True,
        )
        if debug:
            print(f"  ~~ GAME BETWEEN: {self.speaker.id} - {self.hearer.id} ~~")
            print(f"  ~~ CONTEXT: {sorted(self.context)} ~~")
            print(f"  ~~ TOPIC: {self.topic} ~~")
            self.world.print_contextualization(self.topic, self.context)
            print(
                f"  ~~ DISCRIMINATING CATEGORIES: {sorted(self.discriminative_cats, key=len)} ~~"
            )

    def step(self, debug=False):
        """Interaction script of the guessing game"""
        if self.discriminative_cats:
            # arm selection
            # speaker chooses arm ifo topic
            utterance = self.speaker.policy(SPEAKER, self.discriminative_cats)
            # hearer chooses arm ifo utterance
            interpretations = self.hearer.policy(HEARER, utterance)

            if debug:
                print(f" === {self.speaker.id} q-table:")
                self.speaker.print_lexicon()
                print(f" === {self.speaker.id} uttered {utterance}")
                print(f" === {self.hearer.id} q-table:")
                self.hearer.print_lexicon()
                print(f" === {self.hearer.id} interpreted {self.hearer.topic}")

            # evaluate pulls
            if (
                interpretations is None
                or self.hearer.applied_cxn is None
                or (self.hearer.topic and self.hearer.topic != self.speaker.topic)
            ):
                self.hearer.adopt(self.speaker.topic, utterance)  # adopt arms
                self.speaker.communicative_success = False
                self.hearer.communicative_success = False
                print(" ===> FAILURE, hence adopting {utterance} <===")
            else:
                print(" ===> SUCCESS <===")
            # learn based on outcome
            self.speaker.align()
            self.hearer.align()
        else:  # no discriminating categories for the topic
            print(" ===> FAILURE, due to no discrimination <===")
            self.speaker.communicative_success = False
            self.hearer.communicative_success = False
