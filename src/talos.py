"""
Main algorithm.
"""
# Author: Christian Lang <me@christianlang.io>

from .environments.four import Four
from .models.talos_mlp import talos_mlp

import datetime as dt
from tqdm import tqdm


def talos():

    board = Four()
    network = talos_mlp(input_dim = 42, output_dim = 8)
    tree = {}
    history = []

    while True:

        board.new_game()

        game = []
        black = True
        while board.winner is None:

            state = board.get_state()
            if not tuple(state.flatten()) in tree:
                print('New')
                tree[tuple(state.flatten())] = (0, 0, 0)

            actions = network.predict(state.reshape(1, -1))

            game.append([state, search, black])

            board.check()

            exit(0)

    return network, tree
if __name__ == '__main__':
    talos()
