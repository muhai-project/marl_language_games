from environment import Environment

class Experiment():
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward = 0
        self.timesteps = 0
        self.env = Environment(self)
        self.communicative_success = []
        
    def run_experiment(self):
        for i in range(0, self.cfg.EPISODES):
            print(f"\n\n - Episode {i} - reward: {self.reward}")   
            self.env.run_episode()
            reward = self.cfg.REWARD_SUCCESS if self.env.communicative_success else self.cfg.REWARD_FAILURE
            self.communicative_success.append(self.env.communicative_success)
            self.reward += reward
            self.timesteps += 1