from emrl.environment.bng.bng_env import BasicNamingGameEnv
from emrl.environment.gg.gg_env import GuessingGameEnv
from emrl.experiment.monitors import Monitors


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors = Monitors(self)

    def initialize(self):
        self.global_reward = 0
        self.timesteps = 0
        self.env = self.select_env(self.cfg)

    def run_experiment(self):
        for serie in range(self.cfg.SERIES):
            self.initialize()
            for i in range(0, self.cfg.EPISODES):
                print(f"\n\n - Episode {i} - population reward: {self.global_reward}")
                debug = (
                    True
                    if self.cfg.PRINT_EVERY and i % self.cfg.PRINT_EVERY == 0
                    else False
                )
                self.env.reset(debug)
                self.env.step(debug)
                self.record_events(serie)  # monitors
            self.print_debug()

    def record_events(self, trial):
        """Records the event of a trial to the monitor."""
        # communicative success
        self.monitors.record_communicative_success(trial)
        # average lexicon size
        self.monitors.record_lexicon_size(trial)
        # lexicon coherence between interacting agents
        self.monitors.record_lexicon_coherence(trial)
        # lexicon similarity (loetszch version of lexicon coherence)
        self.monitors.record_lexicon_similarity(trial)
        # lexicon change
        self.monitors.record_lexicon_change(trial)
        # avg forms per meaning
        self.monitors.record_forms_per_meaning(trial)
        # avg meanings per form
        self.monitors.record_meanings_per_form(trial)
        # record shared global cumulative reward
        self.global_reward += (
            self.cfg.REWARD_SUCCESS
            if self.env.speaker.communicative_success
            else self.cfg.REWARD_FAILURE
        )
        # record timesteps
        self.timesteps += 1

    def run_competition(self):
        """Runs and generates data for a form-competition graph

        This experiment logs the form competition in a lexicon of a specific agent for a specific meaning.
        """
        self.initialize()
        agent_tracked, object_tracked = 1, 2
        for i in range(0, self.cfg.EPISODES):
            print(f"\n\n - Episode {i} - population reward: {self.global_reward}")
            debug = (
                True
                if self.cfg.PRINT_EVERY and i % self.cfg.PRINT_EVERY == 0
                else False
            )
            self.env.reset(debug)
            self.env.step(debug)
            self.record_competition(i, agent_tracked, object_tracked)

        print("Lexicon of the tracked agent:")
        ag = self.env.population[agent_tracked]
        print(ag)
        ag.print_lexicon()

    def record_competition(self, episode, agent_idx, obj_idx):
        # form competition
        self.monitors.record_form_competition(episode, agent_idx, obj_idx)

    def select_env(self, cfg):
        if self.cfg.ENV == "bng":
            return BasicNamingGameEnv(cfg)
        elif self.cfg.ENV == "gg":
            return GuessingGameEnv(cfg)
        else:
            raise ValueError(f"Given environment {self.cfg.ENV} is not valid!")

    def print_debug(self):
        print("population: ")
        for ag in self.env.population:
            print(ag)
            ag.print_lexicon()
