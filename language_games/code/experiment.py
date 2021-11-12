from bng_env import BasicNamingGameEnv
from gg_env import GuessingGameEnv

from monitors import Monitors

class Experiment():
    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors = Monitors(self)

    def initialize(self):
        self.reward = 0 # TODO reward here doesn't fit, its cumulated comm. success, but interactions are local so no cumulated shared reward
        self.timesteps = 0
        self.env = self.select_env(self.cfg)
        
    def run_experiment(self):
        for serie in range(self.cfg.SERIES): # [LG] - RL literature doesn't do multiple series as it is deemed too expensive
            self.initialize()
            for i in range(0, self.cfg.EPISODES): # [RL] - episode = a single interaction
                print(f"\n\n - Episode {i} - reward: {self.reward}")   
                self.env.reset() # [RL] - corresponds to :before interaction, i.e. pick context, topic and reset slots
                self.env.step() # [RL] - single step episode (bandit)
                self.record_events(serie) # monitors
            self.print_debug()

    def record_events(self, serie):
        """Records the event of a serie to the monitor."""
        # communicative success
        self.monitors.record_communicative_success(serie)
        # average lexicon size
        self.monitors.record_lexicon_size(serie)
        # record cumulative reward
        self.reward += self.cfg.REWARD_SUCCESS if self.env.speaker.communicative_success else self.cfg.REWARD_FAILURE
        # record timesteps
        self.timesteps += 1

    def select_env(self, cfg):
        if self.cfg.ENV == "bng":
            return BasicNamingGameEnv(cfg)
        elif self.cfg.ENV == "gg":
            return GuessingGameEnv(cfg)
        else:
            raise Exception(f"Given environment {self.cfg.ENV} is not valid!")

    def print_debug(self):
        print("population: ")
        for ag in self.env.population:
            print(ag)
            ag.print_lexicon()