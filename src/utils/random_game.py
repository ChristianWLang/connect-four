"""
Random game example.
"""
# Author: Christian Lang <me@christianlang.io>

from ..environment.board import Board

import numpy as np


def random_game():

    board = Board()
    board.new_game()

    black = True
    while board.winner is None:
        
        pos = np.random.randint(7)

        valid_move = False
        try:
            board.move(pos, black = black)
            valid_move = True
        except:
            pos = np.random.randint(7)
            pass

        board.check()

        if black:
            black = False

        else:
            black = True

    return board.record, board.winner
if __name__ == '__main__':
    record, winner = random_game()
