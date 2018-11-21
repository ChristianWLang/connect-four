"""
Random game example.
"""
# Author: Christian Lang <me@christianlang.io>

from ..environment.board import Board

import numpy as np

import time


def random_game():

    board = Board()
    board.new_game()

    black = True
    while board.winner is None:
        
        time.sleep(.5)
        
        pos = np.random.randint(7)

        valid_move = False
        try:
            board.move(pos, black = black)
            valid_move = True
        except:
            pos = np.random.randint(7)
            pass

        board.check()

        screen = board.show_board()
        print(screen)

        if black:
            black = False

        else:
            black = True

    print(board.winner)
    print(board.record)

    return
if __name__ == '__main__':
    random_game()
