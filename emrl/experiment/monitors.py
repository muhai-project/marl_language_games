import os
from collections import defaultdict

from emrl.utils.convert_data import write_measure, write_measure_competition


class Monitors:
    def __init__(self, exp):
        self.exp = exp
        self.monitors = defaultdict(list)

    def add_event_to_trial(self, monitor, trial, event):
        """Adds a new event to a monitor in a given trial."""
        if len(monitor) <= trial:
            monitor.append([event])  # assumes trials are executed sequentially
        else:
            monitor[trial].append(event)

    def write(self, logdir):
        logdir = os.path.join(logdir, "monitors")
        os.makedirs(logdir, exist_ok=True)

        for key, data in self.monitors.items():
            write_measure(data, os.path.join(logdir, key))

    def record_communicative_success(self, trial):
        """Records the success of the current interaction.

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        event = self.exp.env.speaker.communicative_success
        monitor = self.monitors["communicative-success"]
        self.add_event_to_trial(monitor, trial, event)

    def calculate_lexicon_size(self, lexicon):
        if self.exp.cfg.IGNORE_LOW_SA_PAIR:
            filtered_list = list(
                filter(
                    lambda sa_pair: sa_pair.q_val
                    > self.cfg.REWARD_FAILURE + self.cfg.EPSILON_FAILURE,
                    lexicon,
                )
            )
            return len(filtered_list)
        else:
            return len(lexicon)

    def record_lexicon_size(self, trial):
        """Records the average number of words known by the population.

        The number of meaning-form associations in each agent's lexicon is counted and averaged over
        the number of agents.

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        sizes = []
        for agent in self.exp.env.population:
            lex_size = self.calculate_lexicon_size(agent.lexicon)
            sizes.append(lex_size)

        event = int(sum(sizes) / len(sizes))  # average lexicon size
        monitor = self.monitors["lexicon-size"]
        self.add_event_to_trial(monitor, trial, event)

    def lexicon_similarity(self, speaker_lex, hearer_lex):
        """Returns a measure how similar the lexicons of the interacting agents are.

        The degree of lexicon overlap between speaker s and hearer h of the current interaction is computed
        as the fraction of form meaning associations that are shared by speaker and hearer and all words known by
        speaker and hearer.
        A slightly more precise measure for coherence in the population would be to compare the lexicons of all agents.
        But because computing coherence between all pairs of agents would be very costly, we chose to use coherence
        between speaker and hearer as a approximation for population coherence.

        Args:
            speaker_lex (list): lexicon of the speaker
            hearer_lex (list): lexicon of the hearer

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

    def record_lexicon_similarity(self, trial):
        """Records how similar the lexicons of the interacting agents are.

        Martin Loetszch's thesis calls this measure lexicon coherence.

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        speaker_lex, hearer_lex = (
            self.exp.env.speaker.lexicon.q_table,
            self.exp.env.hearer.lexicon.q_table,
        )
        event = self.lexicon_similarity(speaker_lex, hearer_lex)
        monitor = self.monitors["grammar-similarity"]
        self.add_event_to_trial(monitor, trial, event)

    def record_lexicon_coherence(self, trial):
        """Records how coherent the lexicons of the interactings agents are for the topic.

        Coherence is measured by inspecting whether the hearer would produce
        the same utterance for the given topic inside the context (must be measured before alignment!).

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        event = self.exp.env.lexicon_coherence
        monitor = self.monitors["lexicon-coherence"]
        self.add_event_to_trial(monitor, trial, event)

    def record_forms_per_meaning(self, trial):
        """Records the average number of forms associated to each meaning by an agent is
        averaged over all agents in the population

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        avgs = []
        for agent in self.exp.env.population:
            meanings = defaultdict(int)
            for sa_pair in agent.lexicon.q_table:
                meanings[sa_pair.meaning] += 1

            counts = list(meanings.values())
            if counts:
                avg = sum(counts) / len(counts)  # average forms per meaning
            else:
                avg = 0
            avgs.append(avg)

        # average forms per meaning for the population
        if avgs:
            event = sum(avgs) / len(avgs)
        else:
            event = 0
        monitor = self.monitors["forms-per-meaning"]
        self.add_event_to_trial(monitor, trial, event)

    def record_meanings_per_form(self, trial):
        """Records the average number of meanings associated to each form by an agent is
        averaged over all agents in the population

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        avgs = []
        for agent in self.exp.env.population:
            forms = defaultdict(int)
            for sa_pair in agent.lexicon.q_table:
                forms[sa_pair.form] += 1

            counts = list(forms.values())
            if counts:
                avg = sum(counts) / len(counts)  # average meanings per form
            else:
                avg = 0
            avgs.append(avg)

        # average forms per meaning for the population
        if avgs:
            event = sum(avgs) / len(avgs)
        else:
            event = 0
        monitor = self.monitors["meanings-per-form"]
        self.add_event_to_trial(monitor, trial, event)

    def record_lexicon_change(self, trial):
        """Records how stable the agents' lexicons are.

        For each interaction in which either the speaker or hearer add or remove a entry in the q_table,
        a value of 1 is recorded, for all others 0.

        Args:
            trial (int): index denoting which trial the new record belongs to
        """
        event = self.exp.env.lexicon_change
        monitor = self.monitors["lexicon-change"]
        self.add_event_to_trial(monitor, trial, event)

    def add_event_competition(self, monitor, events, episode):
        """Adds competition events to the given monitor.

        This function also logs NIL entries for entries in the monitor that are not tracked anymore.
        For example, when a SA_Pair is deleted from the lexicon, deleted sa_pair will be logged as NIL.

        Args:
            monitor (dict): monitor to which the events are added
            events (list): list of (key, score) tuples
            episode (int): current episode index
        """
        keys = list(monitor.keys())
        for event in events:
            key, val = event
            if key not in monitor:
                [monitor[key].append("NIL") for i in range(episode)]
            monitor[key].append(val)
            if key in keys:
                keys.remove(key)
        for key in keys:
            monitor[key].append("NIL")

    def record_form_competition(self, episode, agent_idx, obj_idx):
        """Records the form competition for the specified object in the lexicon of the specified agent."""
        if "form-competition" not in self.monitors:
            # initialize competition monitor
            self.monitors["form-competition"] = defaultdict(list)

        meaning = self.exp.env.world.objects[obj_idx]
        agent = self.exp.env.population[agent_idx]

        events = []
        for sa_pair in agent.lexicon.q_table:
            if meaning == sa_pair.meaning:
                events.append((sa_pair.form, sa_pair.q_val))

        monitor = self.monitors["form-competition"]
        self.add_event_competition(monitor, events, episode)

    def write_competition(self, logdir):
        logdir = os.path.join(logdir, "monitors")
        os.makedirs(logdir, exist_ok=True)

        for key, data in self.monitors.items():
            write_measure_competition(data, os.path.join(logdir, key))
