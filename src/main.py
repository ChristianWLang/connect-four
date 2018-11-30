"""
Main algorithm.
"""
# Author: Christian Lang <me@christianlang.io>

from .environments.board import Board
from .models.mlp import MLP

import numpy as np


def main(start_greedy = 1,
        end_greedy = .1,
        initial_greedy = 1000,
        anneal_greedy = 5000):

    board = Board()
    network = MLP(input_dim = 84, output_dim = 1)
    history = []
    greedy = start_greedy
    turns_per_game = []

    iteration = 0
    while True:

        board.new_game()
        turns = 0

        game = []
        while board.winner is None:

            state_0 = board.get_state()

            if not board.turn:
                state_0 *= -1

            representation_0 = np.zeros(shape = (42, 2))
            p1 = np.where(state_0.flatten() == 1, 1, 0)
            p2 = np.where(state_0.flatten() == -1, 1, 0)

            representation_0[:, 0] += p1
            representation_0[:, 1] += p2

            if np.random.uniform() < greedy:
                actions = np.random.uniform(0, 1, size = (1, 7))
            else:
                actions = np.zeros(shape = (1, 7))
                legal = board.legal_moves()
                for i in range(len(legal)):
                    if legal[i] == 1:
                        
                        glimpse = board.view_move(i, board.turn)

                        if not board.turn:
                            glimpse *= -1
                        
                        glimpse_representation = np.zeros(shape = (42, 2))
                        p1 = np.where(glimpse.flatten() == 1, 1, 0)
                        p2 = np.where(glimpse.flatten() == -1, 1, 0)
                        
                        glimpse_representation[:, 0] += p1
                        glimpse_representation[:, 1] += p2

                        actions[0][i] = network.predict(glimpse_representation.reshape(1, -1))

            actions = np.where(board.legal_moves() == 1, actions, -1)
            moves = actions.argsort().flatten()[::-1]

            for move in moves:
                try:
                    board.move(move)
                    break
                except:
                    continue
            
            state_1 = board.get_state()

            if not board.turn:
                state_1 *= -1

            representation_1 = np.zeros(shape = (42, 2))
            p1 = np.where(state_1.flatten() == 1, 1, 0)
            p2 = np.where(state_1.flatten() == -1, 1, 0)

            representation_1[:, 0] += p1
            representation_1[:, 1] += p2

            game.append((representation_0, board.turn, representation_1))
            turns += 1
            board.check()

        turns_per_game.append(turns)

        X, y = [], []
        for g in game:
            
            if g[1] == board.winner:
                reward = 1

            else:
                reward = 0
            
            for i in [0, 2]:
                X.append(g[i].reshape(1, -1))
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

        if (iteration >= 50000) & (iteration % 10000 == 0):
            print('Saving Model')
            network.save('model/network0_{}.h5'.format(iteration))

    return
if __name__ == '__main__':
    main()
