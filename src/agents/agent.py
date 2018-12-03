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
        self.search = None
        pass

    def act(self, board, simulations = 800, tau = 1):
        '''
        Returns action distribution, as given by agent.
        '''
    
        if self.search is None:
            self.search = MCTS(board = board)

        else:
            self.search.set_root(board = board)

        for i in range(simulations):

            ## Simulate a game with search.simulate
            node, winner, backprop = self.search.simulate(board = board)
            
            ## Evaluate leaf node with search.evaluate
            winner = self.search.evaluate(current = node,
                    winner = winner,
                    network = self.network,
                    board = board)
            
            ## Update the search tree with search.update
            self.search.backpropagation(current = node,
                    winner = winner,
                    backprop = backprop)

            pass
        
        action = self.search.play(tau = tau)

        self.history.append((board.get_representation(), action, board.turn))

        for edge in self.search.root.edges:
            print(edge.stats)

        return action

    def clear_history(self):
        self.history = []
        return

    def train(self, winner, remove = True):
        X, y = [], []

        for record in self.history:
            X.append(record[0])
            if (winner == True) | (winner == False):
                if record[2] == winner:
                    y.append(np.append(record[1], 1))

                else:
                    y.append(np.append(record[1], 0))

            else:
                y.append(np.append(record[1], winner))

        X = np.vstack(X)
        y = np.vstack(y)

        self.network.train_on_batch(X, y)

        if remove:
            self.history = []
        return

    def get_network(self):
        return self.network
