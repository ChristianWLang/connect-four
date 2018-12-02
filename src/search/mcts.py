"""
MCTS implementation.
"""
# Author: Christian Lang <me@christianlang.io>

import numpy as np


class Node(object):
    def __init__(self, state, turn, board, network):
        self.state = state
        self.id = str(state)
        self.turn = turn
        self.edges = []

        self.__init_edges__(board, network)

        if len(self.edges) > 0:
            self.is_leaf = False
        else:
            self.is_leaf = True

        print(self.state)
        print(self.id)
        print(self.turn)
        print(self.edges)
        print(self.is_leaf)

        exit(0)
        pass

    def __init_edges__(self, board, network):

        priors = network.predict(
                board.get_representation(state = self.state,
                turn = self.turn)
                ).flatten()

        priors, value = priors[:-1], priors[-1]
        
        legal_moves = board.legal_moves(self.state)
        priors *= legal_moves
        priors = np.exp(priors) / np.exp(priors).sum()

        for i in range(len(legal_moves)):
            if legal_moves[i] == 1:
                action = np.zeros(shape = (len(legal_moves),))
                action[i] = 1
                self.edges.append(
                        Edge(state = self.state, 
                            action = action, 
                            turn = self.turn,
                            prior = priors[i])
                        )

        return

class Edge(object):
    def __init__(self, state, action, turn, prior):
        self.state = state
        self.action = action
        self.turn = turn
        self.stats = {
                'N': 0,
                'W': 0,
                'Q': 0,
                'P': prior,
                }
        pass

class MCTS(object):
    def __init__(self,
            board,
            network,
            puct = 1,
            epsilon = .2,
            alpha = .8):
        
        self.root = Node(state = board.get_state(),
                turn = board.turn,
                board = board,
                network = network)
        
        self.tree = {}
        self.add_node(root)
        self.puct = puct
        self.epsilon = epsilon
        self.alpha = alpha
        pass

    def tree_len(self):
        return len(self.tree)

    def add_node(self, node):
        self.tree[node.id] = node

    def simulate(self):

        breadcrumbs = []
        current = self.root

        done = 0
        value = 0

        while not current.is_leaf:

            maxQU = -99999

            if current == self.root:
                epsilon = self.epsilon
                nu = np.random.dirichlet([self.alpha] * len(current.edges))

            else:
                epsilon = 0
                nu = [0] * len(current.edges)

            nb = 0
            for action, edge in current.edges:
                nb += edge.stats['N']

            for idx, (action, edge) in enumerate(current.edges):

                U = self.puct * ((1 - self.epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                        np.sqrt(nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    sim_action = action
                    sim_edge = edge
