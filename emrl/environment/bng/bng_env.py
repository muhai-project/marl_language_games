import logging
import random

import numpy as np

from emrl.environment.bng.bng_agent import HEARER, SPEAKER, Agent
from emrl.utils.invention import make_id


class World:
    """Abstraction class of the part of the environment which handles the shared world."""

    def __init__(self, world_size):
        """Initializes a world of objects."""
        self.objects = [make_id("OBJ") for i in range(world_size)]

    def pick_topic(self, context):
        """Given a list of objects (context) returns at random one of the objects as the topic."""
        return random.sample(context, k=1)[0]

    def pick_context(self, context_min_size, context_max_size):
        """Given a world chooses a subset of the world of objects as the context.

        The size of the context is sampled uniformly at run-time using the given parameters context_min/max_size.
        """
        context_size = np.random.randint(context_min_size, context_max_size + 1)
        return random.sample(self.objects, k=context_size)


class BasicNamingGameEnv:
    """
    Basic naming game environment

    The basic naming game is a referential game where the goal is
    to achieve communicative success between a population of agents.
    In each episode (called a local interaction) of the game,
    two autonomous agents are selected from the population.
    The agents in the episode are assigned a role either of speaker or hearer.

    In this game, the environment (world) consists of n objects represented by symbols.
    In each episode, a subset of the world is selected as the context of the episode.
    One object of the context is then selected by the speaker as the topic.
    (In this environment, the topic is chosen randomly by sampling an object from the context.)

    The speaker produces an utterance, i.e., a word or a string of characters, for the topic.
    When the speaker does not know a word for the topic,
    a new word is created and stored in the lexicon of the agent.
    The hearer then parses the utterance and interprets it to find the associated meaning (i.e. the topic).
    Both when the hearer does not know a word form f or when he pointed to the wrong
    topic (which does not happen in the Naming Game), the speaker will point to the intended topic m.
    The hearer then adopts the new convention by storing the new word.

    The environment corresponds to the version of the 'naming game' problem
    described in Chapter 4 in Lexicon Formation in Autonomous Robots by Loetzsch

    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.world = World(self.cfg.WORLD_SIZE)
        self.population = [Agent(cfg) for i in range(self.cfg.POPULATION_SIZE)]

    def reset(self):
        """Resets the basic naming game environment."""
        # determine interacting agents
        self.speaker, self.hearer = random.sample(self.population, k=2)

        # determine context and topic
        self.context = self.world.pick_context(
            self.cfg.CONTEXT_MIN_SIZE, self.cfg.CONTEXT_MAX_SIZE
        )
        self.topic = self.world.pick_topic(self.context)

        # reset agent
        self.speaker.reset(self.context)
        self.hearer.reset(self.context)

        # logging
        self.lexicon_change = False
        self.lexicon_coherence = False

    def step(self, idx):
        """Interaction script of the basic naming game

        Args:
            idx (int): denotes the ith interaction in the environment
        """
        # speaker chooses action ifo topic
        utterance, self.lexicon_change = self.speaker.policy(SPEAKER, self.topic)
        # hearer chooses action ifo utterance
        interpretation = self.hearer.policy(HEARER, utterance)

        hearer_utterance = self.hearer.produce_as_hearer(self.topic)  # monitoring
        self.lexicon_coherence = hearer_utterance == utterance  # monitoring

        # evaluate pulls
        if interpretation is None or interpretation != self.topic:
            self.lexicon_change = True  # monitoring
            self.hearer.adopt(self.topic, utterance)
            self.speaker.communicative_success = False
            self.hearer.communicative_success = False

        # learn based on outcome
        self.speaker.align()
        self.hearer.align()

        # debug interactions
        if self.cfg.PRINT_EVERY and idx % self.cfg.PRINT_EVERY == 0:
            self.print_example_interaction(idx, utterance, interpretation)

    def print_example_interaction(self, idx, utterance, interpretation):
        logging.debug(f"\n\n- Episode {idx}")
        logging.debug(f" ~~ GAME BETWEEN: {self.speaker.id} - {self.hearer.id} ~~")
        logging.debug(f" ~~ TOPIC: {self.topic} ~~")
        logging.debug(f" === {self.speaker.id} q-table:")
        logging.debug(self.speaker.lexicon)
        logging.debug(f" === {self.speaker.id} uttered {utterance}")
        logging.debug(f" === {self.hearer.id} q-table:")
        logging.debug(self.hearer.lexicon)
        logging.debug(f" === {self.hearer.id} interpreted {interpretation}")
        if self.speaker.communicative_success:
            logging.debug(f" ===> FAILURE, hence adopting {utterance} <===")
        else:
            logging.debug(" ===> SUCCESS <===")
