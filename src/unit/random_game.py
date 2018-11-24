"""
Random game example.
"""
# Author: Christian Lang <me@christianlang.io>

from ..environment.four import Four

import numpy as np

import time


def random_game():

    board = Four()
    board.new_game()

    black = True
    while board.winner is None:

        screen = board.show_board()
        print(screen)
        
        time.sleep(1)
        
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

    screen = board.show_board()
    print(screen)
    print(board.winner)

    return
if __name__ == '__main__':
    random_game()
