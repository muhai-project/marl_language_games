import numpy as np

from bng_agent import Agent, HEARER, SPEAKER
from environment import Environment
from utils import make_id

class BasicNamingGameEnv(Environment):
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
    def __init__(self, experiment):
        super().__init__(experiment)
        # The environment (world) consists of a set of objects.
        self.world = [make_id("OBJ") for i in range(self.cfg.WORLD_SIZE)]
        self.population = [Agent(experiment) for i in range(self.cfg.WORLD_SIZE)]


    def reset(self):
        """Resets the basic naming game environment."""
        self.topic = np.random.choice(self.world)
        print(f" === topic: {self.topic}")
        # choose interacting agents
        self.speaker, self.hearer = np.random.choice(self.population, 
                                                   size=2, 
                                                   replace=False)
        # reset agent
        self.speaker.applied_cxn, self.hearer.applied_cxn = None, None
        self.speaker.communicative_success, self.hearer.communicative_success = True, True
        
    def step(self):
        """Interaction script of the basic naming game"""
        # arm selection
        utterance = self.speaker.policy(SPEAKER, self.topic) # [RL] - speaker chooses arm (construction) ifo topic
        interpretation = self.hearer.policy(HEARER, utterance) # [RL] - hearer chooses arm (construction) ifo utterance
        
        # evaluate pulls
        print(f" === {self.speaker.id} uttered {utterance}")
        print(f" === {self.hearer.id} interpreted {interpretation}")
        if interpretation == None or interpretation != self.topic:
            self.hearer.adopt(self.topic, utterance) # [LG] - adoption of the unseen state/action pair
            self.speaker.communicative_success = False
            self.hearer.communicative_success = False
            print(f" ===> FAILURE, hence adopting {utterance} <===")
        else:
            print(f" ===> SUCCESS <===")
        
        # learn based on outcome
        self.speaker.align() # [LG] - value iteration
        self.hearer.align() # [LG] - value iteration