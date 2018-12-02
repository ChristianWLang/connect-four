"""
Generalized RL Agent.
"""
# Author: Christian Lang <me@christianlang.io>

import numpy as np
from ..search.mcts import MCTS


class Agent(object):
    
    def __init__(self, start_greedy, end_greedy, anneal_start, anneal_end, network = None):
        
        self.start_greedy = start_greedy
        self.end_greedy = end_greedy
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.network = network
        self.history = []
        pass

    def act(self, board, simulations = 800):
        '''
        Returns action distribution, as given by agent.
        '''

        search = MCTS(board = board, 
                network = self.network)

        for i in range(simulations):
            ## Simulate a game with search.simulate
            ## If needed, evaluate terminal state
            ## Update the search tree with search.update

        actions = np.random.uniform(0, 1, size = (1, 7))
        value = np.random.uniform()

        self.history.append((board.get_representation(), actions, board.turn))

        return actions, value

    def clear_history(self):
        self.history = []
        return

    def train(self, remove = True):
        if remove:
            self.history = []
        return

    def get_network(self):
        return self.network
