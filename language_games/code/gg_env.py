import random
import numpy as np
from collections import defaultdict

from gg_agent import Agent, HEARER, SPEAKER
from environment import Environment
from utils import make_id


class World():
    def __init__(self, world_size, amount_cats, cats_per_obj):
        """Initializes a world of objects represented by a set of categories."""
        self.cats = [make_id("C") for i in range(amount_cats)]
        self.objs = defaultdict(list)
        for _ in range(world_size):
            self.objs[make_id("OBJ")] = random.sample(self.cats, cats_per_obj)
        self.objects = list(self.objs.keys())

    def pick_topic(self, context):
        return np.random.choice(context)

    def pick_context(self, context_min_size, context_max_size):
        context_size = np.random.randint(context_min_size, context_max_size+1)
        return np.random.choice(self.objects, size=context_size, replace=False)

    def get_categories(self, obj):
        return self.objs[obj]

    def conceptualize(self, topic, context):
        other_objects = [obj for obj in context if obj != topic]
        discr_cats = set(self.objs[topic])
        for obj in sorted(list(other_objects)):
            discr_cats -= set(self.objs[obj])
        return list(discr_cats)


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
    In order to do so, the speaker internally needs to figure out which categories of the topic are discriminative
    in relation to set of objects of the context.
    
    The environment corresponds to the version of the 'guessing game' problem
    described in Chapter 5 of 'Lexicon Formation in Autonomous Robots' by Loetzsch.
    """
    def __init__(self, experiment):
        super().__init__(experiment)
        self.world = World(self.cfg.WORLD_SIZE, self.cfg.AMOUNT_CATEGORIES, self.cfg.CATEGORIES_PER_OBJECT)
        self.context, self.topic, self.discriminative_cats = None, None, None
        self.speaker, self.hearer = None, None
        self.population = [Agent(self.cfg, self.world) for i in range(self.cfg.POPULATION_SIZE)]

    def reset(self):
        """Resets the guessing game environment."""
        # choose interacting agents
        self.speaker, self.hearer = np.random.choice(self.population,
                                                   size=2,
                                                   replace=False)
        
        self.context = self.world.pick_context(self.cfg.CONTEXT_MIN_SIZE, self.cfg.CONTEXT_MAX_SIZE)
        self.topic = self.world.pick_topic(self.context)
        self.discriminative_cats = self.world.conceptualize(self.topic, self.context)  

        # reset agent
        self.speaker.context, self.hearer.context = self.context, self.context
        self.speaker.applied_cxn, self.hearer.applied_cxn = None, None
        self.speaker.correct_path, self.hearer.correct_path = None, None
        self.speaker.parsed_lexs, self.hearer.parsed_lexs = None, None
        self.speaker.topic, self.hearer.topic = self.topic, None
        self.speaker.communicative_success, self.hearer.communicative_success = True, True
        self.correct_path = False
        print(f"  ~~ GAME BETWEEN: {self.speaker.id} - {self.hearer.id} ~~")
        print(f"  ~~ TOPIC: {self.topic}, {self.discriminative_cats} ~~")

    def step(self):
        """Interaction script of the guessing game"""
        if self.discriminative_cats:
            # arm selection
            utterance = self.speaker.policy(SPEAKER, self.discriminative_cats) # speaker chooses arm ifo topic
            interpretations = self.hearer.policy(HEARER, utterance) # hearer chooses arm ifo utterance

            # evaluate pulls
            print(f" === {self.speaker.id} uttered {utterance}")
            print(f" === {self.hearer.id} interpreted {self.hearer.topic}")
            if interpretations is None or self.hearer.applied_cxn is None or (self.hearer.topic and self.hearer.topic != self.speaker.topic):
                self.hearer.adopt(self.speaker.topic, utterance) # adopt arms
                self.speaker.communicative_success = False
                self.hearer.communicative_success = False
                print(f" ===> FAILURE, hence adopting {utterance} <===")
            else:
                print(f" ===> SUCCESS <===")
            # learn based on outcome
            self.speaker.align()
            self.hearer.align()
        else: # no discriminating categories for the topic
            print(f" ===> FAILURE, due to no discrimination <===")
            self.speaker.communicative_success = False
            self.hearer.communicative_success = False
