from environment import Environment

class GuessingGameEnv(Environment):
    """
    Guessing game environment #TODO

    The guessing game is a referential game where the goal is 
    to achieve communicative success between a population of agents.
    In each episode (called a local interaction) of the game, 
    two autonomous agents are selected from the population.
    The agents in the episode are assigned a role either of speaker or hearer.

    In this game, the environment (world) consists of n objects represented by a set of categories.
    In each episode, a subset of the world is selected as the context of the episode.
    One object of the context is then selected by the speaker as the topic.
    (In this environment, the topic is chosen randomly by sampling an object from the context.)
    The speaker's goal is to successfully communicate to the hearer the topic.
    To do so, the speaker internally needs to figure out which categories of the topic are discriminative 
    in relation to set of objects of the context. 

    The speaker produces an utterance, i.e., a word or a string of characters, for the topic.
    When the speaker does not know a word for the topic, 
    a new word is created and stored in the lexicon of the agent. 
    The hearer then parses the utterance and interprets it to find the associated meaning (i.e. the topic).
    Both when the hearer does not know a word form f or when he pointed to the wrong
    topic (which does not happen in the Naming Game), the speaker will point to the intended topic m. 
    The hearer then adopts the new convention by storing the new word.
    
    The environment corresponds to the version of the 'guessing game' problem
    described in Chapter 5 in Lexicon Formation in Autonomous Robots by Loetzsch.
    
    """
    def __init__(self, experiment):
        super().__init__(experiment)
        self.population = [Agent(experiment) for i in range(self.cfg.WORLD_SIZE)]


    def reset(self):
        """Resets the guessing game environment."""
        pass
        
        
    def step(self):
        """Interaction script of the basic naming game"""
        # arm selection

        # evaluate pulls
        
        # learn based on outcome
        pass
