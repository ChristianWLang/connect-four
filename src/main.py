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

    network = MLP(input_dim = 84, output_dim = 8)
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

        while winner is None:

            actions, value = agent.act(board)
            board.move(actions)
            winner = board.check()

        agent.train()
        iteration += 1

        if (iteration % 50 == 0):
            print(iteration)

        if (iteration >= 1000) & (iteration % 1000 == 0):
            print('Saving Model')
            network = agent.get_network()
            network.save('model/network0_{}.h5'.format(iteration))

    return
if __name__ == '__main__':
    main()
