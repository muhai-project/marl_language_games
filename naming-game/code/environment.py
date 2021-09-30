import numpy as np

from agent import Agent
from utils import make_id

from agent import HEARER, SPEAKER

class Environment():
    def __init__(self, experiment):
        self.cfg = experiment.cfg
        # The environment (world) consists of a set of objects.
        self.world = [make_id("OBJ") for i in range(self.cfg.WORLD_SIZE)]
        self.population = [Agent(experiment) for i in range(self.cfg.WORLD_SIZE)]

        # episodic states
        self.topic = None
        self.communicative_success = True
        
        self.interacting_agents = None
        self.speaker, self.hearer = None, None
        
        
    def reset(self):
        self.topic = np.random.choice(self.world)
        self.communicative_success = True

        # choose a 
        self.interacting_agents = np.random.choice(self.population, 
                                                   size=2, 
                                                   replace=False)
        self.speaker, self.hearer = self.interacting_agents[0], self.interacting_agents[1]
        # reset agent
        self.speaker.role, self.hearer.role = SPEAKER, HEARER
        self.speaker.applied_cxn, self.hearer.applied_cxn = None, None
        
    def run_episode(self):
        """Interaction script"""

        self.reset()
        # arm selection
        utterance = self.speaker.policy(self.topic) # speaker chooses arm ifo topic
        interpretation = self.hearer.policy(utterance) # hearer chooses arm ifo utterance
        
        # evaluate pulls
        if interpretation == None or interpretation != self.topic:
            self.hearer.adopt(self.topic, utterance) # adopt arm
            self.communicative_success = False
        reward = self.cfg.REWARD_SUCCESS if self.communicative_success else self.cfg.REWARD_FAILURE

        # learn based on outcome
        for agent in self.interacting_agents:
            agent.align(reward)