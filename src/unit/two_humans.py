"""
Random game example.
"""
# Author: Christian Lang <me@christianlang.io>

from ..environments.board import Board

import numpy as np

import time


def two_humans():

    board = Board()
    board.new_game()
    
    winner = None

    while winner is None:
        
        screen = board.show_board()
        print(screen)

        if board.turn:
            whos = 'X'

        else:
            whos = 'O'

        pos = int(input("{}'s turn: ".format(whos)))

        move = np.zeros(7)
        print(move)
        move[pos] = 1

        print(move)

        try:
            board.move(move)
        except:
            pos = int(input("Invalid move, try again: ".format(whos)))

            move = np.zeros(7)
            move[pos] = 1

            pass

        winner = board.check()

    screen = board.show_board()
    print(screen)
    print(winner)

    return
if __name__ == '__main__':
    two_humans()
