from basic_naming_game import BasicNamingGameEnv
from guessing_game import GuessingGameEnv

from monitors import Monitors

class Experiment():
    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors = Monitors(self)

    def initialize(self):
        self.reward = 0
        self.timesteps = 0
        self.env = self.select_env()
        
    def run_experiment(self):
        # self.env.two_dialect_setting(self)
        for serie in range(self.cfg.SERIES):
            self.initialize()
            for i in range(0, self.cfg.EPISODES):
                print(f"\n\n - Episode {i} - reward: {self.reward}")   
                self.env.reset()
                self.env.step()
                self.record_events(serie)
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


    def select_env(self):
        if self.cfg.ENV == "bng":
            return BasicNamingGameEnv(self)
        elif self.cfg.ENV == "gg":
            return GuessingGameEnv(self)
        else:
            raise Exception(f"Given environment {self.cfg.ENV} is not valid!")

    def print_debug(self):
        print("population: ")
        for ag in self.env.population:
            print(ag)
            ag.print_lexicon()