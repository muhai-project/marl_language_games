import os
from collections import defaultdict

from plot import write_measure

class Monitors():
    def __init__(self, exp):
        self.exp = exp
        self.monitors = defaultdict(list)

    def add_event_to_serie(self, monitor, serie, event):
        """Adds a new event to a monitor in a given serie."""
        if len(monitor) <= serie: 
            monitor.append([event]) # assumes series are executed sequentially
        else:
            monitor[serie].append(event)


    def record_communicative_success(self, serie):
        event = self.exp.env.speaker.communicative_success
        monitor = self.monitors["communicative_success"]
        self.add_event_to_serie(monitor, serie, event)

    def record_lexicon_size(self, serie):
        """Records the average lexicon size of the population of the experiment"""
        sizes = []
        for agent in self.exp.env.population:
            sizes.append(len(agent.lexicon))
            
        event = int(sum(sizes)/len(sizes)) # average lexicon size
        monitor = self.monitors["lexicon_size"]
        self.add_event_to_serie(monitor, serie, event)

    def write(self, logdir):
        logdir = os.path.join(logdir, "monitors")
        os.makedirs(logdir, exist_ok=True)

        for key, data in self.monitors.items():
            write_measure(data, os.path.join(logdir, key))

