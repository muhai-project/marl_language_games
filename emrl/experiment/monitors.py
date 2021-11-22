import os
from collections import defaultdict

import numpy as np

from emrl.utils.plot import write_measure


class Monitors:
    def __init__(self, exp):
        self.exp = exp
        self.monitors = defaultdict(list)

    def add_event_to_serie(self, monitor, serie, event):
        """Adds a new event to a monitor in a given serie."""
        if len(monitor) <= serie:
            monitor.append([event])  # assumes series are executed sequentially
        else:
            monitor[serie].append(event)

    def write(self, logdir):
        logdir = os.path.join(logdir, "monitors")
        os.makedirs(logdir, exist_ok=True)

        for key, data in self.monitors.items():
            write_measure(data, os.path.join(logdir, key))

    def record_communicative_success(self, serie):
        """Records the success of the current interaction.

        Args:
            serie (int): index denoting which trial the new record belongs to
        """
        event = self.exp.env.speaker.communicative_success
        monitor = self.monitors["communicative-success"]
        self.add_event_to_serie(monitor, serie, event)

    def record_lexicon_size(self, serie):
        """Records the average number of words known by the population.

        The number of meaning-form associations in each agent's lexicon is counted and averaged over
        the number of agents.

        Args:
            serie (int): index denoting which trial the new record belongs to
        """
        sizes = []
        for agent in self.exp.env.population:
            sizes.append(len(agent.lexicon))

        event = int(sum(sizes) / len(sizes))  # average lexicon size
        monitor = self.monitors["lexicon-size"]
        self.add_event_to_serie(monitor, serie, event)

    def lexicon_coherence(self, speaker_lex, hearer_lex):
        """Returns a measure how similar the lexicons of the interacting agents are.

        The degree of lexicon overlap between speaker s and hearer h of the current interaction is computed
        as the fraction of form meaning associations that are shared by speaker and hearer and all words known by
        speaker and hearer.
        A slightly more precise measure for coherence in the population would be to compare the lexicons of all agents.
        But because computing coherence between all pairs of agents would be very costly, we chose to use coherence
        between speaker and hearer as a approximation for population coherence.

        Args:
            experiment (Experiment): The experiment object containing the local interaction.

        Returns:
            float: a number between [0, 1] denoting the coherence of the given lexicons
        """
        intersected_lex = set(speaker_lex).intersection(set(hearer_lex))

        coherence = 0
        if speaker_lex or hearer_lex:  # to avoid divide by zero
            coherence = (2 * len(intersected_lex)) / (
                len(speaker_lex) + len(hearer_lex)
            )
        return coherence

    def record_lexicon_coherence(self, serie):
        """Records how similar the lexicons of the interacting agents are.

        Args:
            serie (int): index denoting which trial the new record belongs to
        """
        speaker_lex, hearer_lex = (
            self.exp.env.speaker.lexicon.q_table,
            self.exp.env.hearer.lexicon.q_table,
        )
        event = self.lexicon_coherence(speaker_lex, hearer_lex)
        monitor = self.monitors["lexicon-coherence"]
        self.add_event_to_serie(monitor, serie, event)

    def record_forms_per_meaning(self, serie):
        """Records the average number of forms associated to each meaning by an agent is
        averaged over all agents in the population

        Args:
            serie (int): index denoting which trial the new record belongs to
        """
        avgs = []
        for agent in self.exp.env.population:
            meanings = defaultdict(int)
            for cxn in agent.lexicon.q_table:
                meanings[cxn.meaning] += 1

            counts = list(meanings.values())
            if counts:
                avg = np.average(counts)  # average forms per meaning
            else:
                avg = 0
            avgs.append(avg)

        # average forms per meaning for the population
        if avgs:
            event = np.average(avgs)
        else:
            event = 0
        monitor = self.monitors["forms-per-meaning"]
        self.add_event_to_serie(monitor, serie, event)

    def record_lexicon_change(self, serie):
        """Records how stable the agents' lexicons are.

        For each interaction in which either the speaker or hearer add or remove a entry in the q_table,
        a value of 1 is recorded, for all others 0.

        Args:
            serie (int): index denoting which trial the new record belongs to
        """
        event = self.exp.env.lexicon_change
        monitor = self.monitors["lexicon-change"]
        self.add_event_to_serie(monitor, serie, event)
