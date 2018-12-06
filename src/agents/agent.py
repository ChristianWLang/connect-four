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
        self.game = []
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

        self.game.append((board.get_representation(), action, board.turn))

        for edge in self.search.root.edges:
            print(edge.stats)

        return action

    def clear_history(self):
        self.history = []
        return

    def clear_game(self):
        self.game = []
        return

    def store(self, winner, remove = True):
        
        for record in self.game:
            if (winner == True) | (winner == False):
                if record[2] == winner:
                    self.history.append((record[0], record[1], 1))

                else:
                    self.history.append((record[0], record[1], -1))

            else:
                self.history.append((record[0], record[1], winner))

        if remove:
            self.clear_game()

        return

    def train(self, batch = 2048, remove = False):

        states = np.random.choice(len(self.history), size = batch)

        X, y1, y2 = [], [], []
        for state in states:
            X.append(self.history[state][0])
            y1.append(self.history[state][1])
            y2.append(self.history[state][2])

        X = np.vstack(X)
        y1 = np.vstack(y1)
        y2 = np.vstack(y2)

        self.network.train_on_batch(X, [y1, y2])

        if remove:
            self.clear_history()

        return

    def evaluate(self, batch_size = 32):

        X, y1, y2 = [], [], []
        for hist in self.history:
            X.append(hist[0])
            y1.append(hist[1])
            y2.append(hist[2])

        X = np.vstack(X)
        y1 = np.vstack(y1)
        y2 = np.vstack(y2)

        loss = self.network.evaluate(X, [y1, y2], batch_size = batch_size)

        return loss

    def get_network(self):
        return self.network
