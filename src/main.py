"""
Main algorithm.
"""
# Author: Christian Lang <me@christianlang.io>

from .environments.board import Board
from .models.mlp import MLP

import numpy as np


def main(start_greedy = 1,
        end_greedy = .1,
        initial_greedy = 5000,
        anneal_greedy = 25000):

    board = Board()
    network = MLP(input_dim = 84, output_dim = 7)
    history = []
    greedy = start_greedy
    turns_per_game = []

    iteration = 0
    while True:

        board.new_game()
        turns = 0

        game = []
        while board.winner is None:

            state = board.get_state()

            if not board.turn:
                state *= -1

            representation = np.zeros(shape = (42, 2))
            p1 = np.where(state.flatten() == 1, 1, 0)
            p2 = np.where(state.flatten() == -1, 1, 0)

            representation[:, 0] += p1
            representation[:, 1] += p2

            if np.random.uniform() < greedy:
                actions = np.random.uniform(-1, 1, size = (1, 7))
            else:
                actions = network.predict(representation.reshape(1, -1))

            actions = np.where(board.legal_moves() == 1, actions, -1)
            moves = actions.argsort().flatten()[::-1]

            for move in moves:
                try:
                    board.move(move)
                    break
                except:
                    continue

            game.append((representation, actions, board.turn))
            turns += 1
            board.check()

        turns_per_game.append(turns)

        X, y = [], []
        for g in game:

            reward = np.zeros(shape = (7,))
            
            if g[2] == board.winner:
                reward[g[1].argmax()] = 1

            else:
                reward[g[1].argmax()] = -1
            
            X.append(g[0].reshape(1, -1))
            y.append(reward)

        X, y = np.vstack(X), np.vstack(y)

        network.train_on_batch(X, y)
        iteration += 1

        if (iteration > initial_greedy) & (iteration < initial_greedy + anneal_greedy):
            greedy -= (start_greedy - end_greedy) / anneal_greedy

        if iteration > (initial_greedy + anneal_greedy):
            greedy = end_greedy
        
        if iteration % 1000 == 0:
            print('Iteration : {} | Greedy : {} | TPG : {}'.format(iteration,
                greedy,
                sum(turns_per_game[-100:]) / 100))

        if (iteration > 50000) & (iteration % 10000 == 0):
            print('Saving Model')
            network.save('model/network.h5')

    return
if __name__ == '__main__':
    main()
