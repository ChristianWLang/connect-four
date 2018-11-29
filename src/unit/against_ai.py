"""
Play against trained model.
"""
# Author: Christian Lang <me@christianlang.io>

from tensorflow import keras
from ..environments.board import Board

from ..models.mlp import _loss
import numpy as np


def against_ai():

    first = input('Go first? (y/n) : ')
    if first.startswith('y'):
        first = True

    else:
        first = False

    board = Board()
    network = keras.models.load_model('model/network.h5', custom_objects = {'_loss': _loss})

    board.new_game()
    while board.winner is None:

        print(board.show_board())

        if first == board.turn:
            moves = input('Move (0 - 6) : ')
            try:
                moves = int(moves)
            except:
                moves = int(input('Move (0 - 6) [int] : '))

            if (moves < 1) | (moves > 7):
                moves = int(input('Move BETWEEN (0 - 6) [int] : '))

            if board.legal_moves()[moves] == 0:
                moves = int(input('Invalid Move : '))

            board.move(moves)

        else:

            state = board.get_state()

            if not board.turn:
                state *= -1

            representation = np.zeros(shape = (42, 2))
            p1 = np.where(state.flatten() == 1, 1, 0)
            p2 = np.where(state.flatten() == -1, 1, 0)

            representation[:, 0] += p1
            representation[:, 1] += p2

            actions = network.predict(representation.reshape(1, -1))

            actions = np.where(board.legal_moves() == 1, actions, -1)
            moves = actions.argsort().flatten()[::-1]

            for move in moves:
                try:
                    board.move(move)
                    break
                except:
                    continue

        board.check()

    print(board.show_board())
    if board.winner:
        print('X wins!')

    else:
        print('O wins!')

    return
if __name__ == '__main__':
    against_ai()
