class Environment():
    """Base class for language game environments."""
    def __init__(self, experiment):
        self.cfg = experiment.cfg
        self.world = None
        self.context = None
        self.population = None
        self.topic = None
        self.speaker, self.hearer = None, None

    def reset(self):
        pass

    def step(self):
        pass