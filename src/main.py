"""
Main algorithm.
"""
# Author: Christian Lang <me@christianlang.io>

from .environments.board import Board
from .agents.agent import Agent
from .models.mlp import MLP

import numpy as np


def main(start_greedy = 1,
        end_greedy = .1,
        anneal_start = 1000,
        anneal_end = 5000):

    network = MLP(input_dim = 84)
    board = Board()
    agent = Agent(start_greedy = start_greedy,
            end_greedy = end_greedy,
            anneal_start = anneal_start,
            anneal_end = anneal_end,
            network = network)

    iteration = 0
    while True:

        board.new_game()
        winner = None

        turn = 1
        tau = 1
        while winner is None:

            print(board.show_board())

            if turn > 16:
                tau = .1
            actions = agent.act(board, simulations = 800, tau = tau)
            board.move(actions)
            winner = board.check()

            turn += 1

        print(board.show_board())
        
        iteration += 1
        
        agent.store(winner = winner)

        agent.train()
        loss = agent.evaluate()
        
        print('{} | {}'.format(iteration, loss))

        if (iteration % 10 == 0):
            print(iteration)

        if (iteration >= 10) & (iteration % 10 == 0):
            print('Saving Model')
            network = agent.get_network()
            network.save('model/network0_{}.h5'.format(iteration))

    return
if __name__ == '__main__':
    main()
