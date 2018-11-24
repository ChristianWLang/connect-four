"""
Connect-Four environment.
"""
# Author: Christian Lang <me@christialang.io>

from .base import Base
import numpy as np


class Four(Base):
    def __init__(self):
        self.record = []
        self.winner = None
        self.key = {'Black': True, 'Red': False}
        self.pieces = {'Black': 'x', 'Red': 'o'}
        self.numeric = {'Black': 1, 'Red': -1}
        super().__init__()
        pass

    def get_tree(self):
        tree = {}

    def show_board(self):
        state_str = np.flip(np.where(self.state == 0, '-', np.where(self.state == 1, 'x', 'o')), axis = 0)
        return state_str

    def new_game(self):
        self.state = np.zeros(shape = (6, 7))
        self.record = []
        self.winner = None
        return

    def move(self, pos, black = True):

        state_0 = np.copy(self.state)

        pieces = int(np.abs(self.state[:, pos]).sum())

        assert pieces < 6

        if black:
            self.state[pieces, pos] = 1

        else:
            self.state[pieces, pos] = -1

        state_1 = np.copy(self.state)

        self.record.append((state_0, pos, black, state_1))

        return

    def check(self):
        for i in range(4):
            
            value = self.state[:, i:i + 4].sum(axis = 1)

            if len(value[value == 4]) > 0:
                self.winner = True
                break

            if len(value[value == -4]) > 0:
                self.winner = False
                break

        for i in range(3):

            value = self.state[i:i + 4, :].sum(axis = 0)

            if len(value[value == 4]) > 0:
                self.winner = True
                break

            if len(value[value == -4]) > 0:
                self.winner = False
                break

        for y in range(self.state.shape[1] - 3):
            for x in range(self.state.shape[0] - 3):

                value = np.array([self.state[x, y],
                    self.state[x+1, y+1],
                    self.state[x+2, y+2],
                    self.state[x+3, y+3]]).sum()

                if value == 4:
                    self.winner = True
                    break

                if value == -4:
                    self.winner = False
                    break

        flipped = np.flip(self.state, axis = -1)
        for y in range(self.state.shape[1] - 3):
            for x in range(self.state.shape[0] - 3):

                value = np.array([flipped[x, y],
                    flipped[x+1, y+1],
                    flipped[x+2, y+2],
                    flipped[x+3, y+3]]).sum()

                if value == 4:
                    self.winner = True
                    break

                if value == -4:
                    self.winner = False
                    break

        if np.abs(self.state).sum() == 42:
            self.winner = 0

        return
