import logging

from tqdm import tqdm

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
        for trial in range(self.cfg.TRIALS):
            logging.info(f" == Experiment trial {trial+1}/{self.cfg.TRIALS} ==")
            self.initialize()
            for i in tqdm(range(0, self.cfg.EPISODES)):
                self.env.reset()
                self.env.step(i)
                self.record_events(trial)  # monitors
            self.log_state_of_lexicons(self.env.population)

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
        for i in tqdm(range(0, self.cfg.EPISODES)):
            logging.debug(
                f"\n\n - Episode {i} - population reward: {self.global_reward}"
            )
            self.env.reset()
            self.env.step(i)
            self.record_competition(i, agent_tracked, object_tracked)

        self.log_state_of_lexicons([self.env.population[agent_tracked]])
        unique_forms = list(self.monitors.monitors["form-competition"].keys())
        logging.info(
            f" Experiment with {len(unique_forms)} unique forms, namely: {unique_forms}"
        )

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

    def log_state_of_lexicons(self, agents):
        logging.debug("State of the lexicons at the end of the experiment: ")
        for ag in agents:
            logging.debug(ag)
            logging.debug(f"\n {ag.lexicon}")
