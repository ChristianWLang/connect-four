"""
MCTS implementation.
"""
# Author: Christian Lang <me@christianlang.io>

import numpy as np


class Node(object):
    def __init__(self, state, turn):
        self.state = state
        self.id = str(state)
        self.turn = turn
        self.edges = []

        pass

    def is_leaf(self):
        if len(self.edges) > 0:
            return False

        else:
            return True

    def expand(self, board, network):

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

        return value

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
            puct = 1,
            epsilon = .2,
            alpha = .8):
        
        self.root = Node(state = board.get_state(),
                turn = board.turn)
        
        self.tree = {}
        self.add_node(self.root)
        self.puct = puct
        self.epsilon = epsilon
        self.alpha = alpha
        pass

    def tree_len(self):
        return len(self.tree)

    def add_node(self, node):
        if not node.id in self.tree:
            self.tree[node.id] = node

    def set_root(self, board):
        self.root = Node(state = board.get_state(),
                turn = board.turn)

        self.add_node(self.root)
        return

    def simulate(self, board):

        backprop = []
        current = self.root

        winner = None

        while not current.is_leaf():

            maxQU = -10e10

            if current == self.root:
                epsilon = self.epsilon
                nu = np.random.dirichlet([self.alpha] * len(current.edges))

            else:
                epsilon = 0
                nu = [0] * len(current.edges)

            nb = 0
            for edge in current.edges:
                nb += edge.stats['N']

            for idx, edge in enumerate(current.edges):

                U = self.puct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                        (np.sqrt(nb) / (1 + edge.stats['N']))

                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    sim_action = edge.action
                    sim_edge = edge

            new_state, new_turn = board.view_move(actions = sim_action,
                    turn = current.turn,
                    state = current.state)
            winner = board.check(state = new_state)

            if str(new_state) in self.tree:
                current = self.tree[str(new_state)]

            else:
                current = Node(state = new_state,
                        turn = new_turn)

            backprop.append(sim_edge)

            if winner is not None:
                break

        return current, winner, backprop

    def evaluate(self, current, winner, network, board):

        if winner is None:

            winner = current.expand(board = board, network = network)
        
        self.add_node(current)

        return winner

    def backpropagation(self, current, winner, backprop):

        for edge in backprop:

            edge_player = edge.turn

            if (winner == True) | (winner == False):
                if edge_player == winner:
                    value_update = 1
                else:
                    value_update = 0

            else:
                if edge_player == current.turn:
                    value_update = winner
                else:
                    value_update = 1 - winner

            edge.stats['N'] += 1
            edge.stats['W'] += value_update
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

        return

    def play(self, tau):

        actions = []
        for edge in self.root.edges:
            actions.append(edge.stats['N'] ** (1/tau))

        actions = np.array(actions)
        actions = actions / actions.sum()

        choice = np.random.choice(np.arange(len(actions)), p = actions)

        action = self.root.edges[choice].action

        return action
