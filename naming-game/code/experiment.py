from monitors import Monitors

class Experiment():
    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors = Monitors(self)
        self.reward = 0
        self.timesteps = 0
        self.env = Environment(self)
        self.communicative_success = []
        
    def run_experiment(self):
        for i in range(0, self.cfg.EPISODES):
            print(f"\n\n - Episode {i} - reward: {self.reward}")   
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