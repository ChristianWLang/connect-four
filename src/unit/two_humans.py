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

        if black:
            whos = 'X'

        else:
            whos = 'O'

        pos = int(input("{}'s turn: ".format(whos)))

        valid_move = False
        try:
            board.move(pos, black = black)
            valid_move = True
        except:
            pos = int(input("Invalid move, try again: ".format(whos)))
            pass

        board.check()

        screen = board.show_board()
        print(screen)

        if black:
            black = False

        else:
            black = True

    print(board.winner)

    return
if __name__ == '__main__':
    random_game()
