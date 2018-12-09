"""
Connect-Four environment.
"""
# Author: Christian Lang <me@christialang.io>

import numpy as np


class Board(object):

    def __init__(self):

        self.record = []
        self.turn = True
        self.state = None
        self.key = {'Black': True, 'Red': False}
        self.pieces = {'Black': 'x', 'Red': 'o'}
        self.numeric = {'Black': 1, 'Red': -1}
        pass
    
    def get_state(self):
        '''
        Returns copy of state.
        '''

        return np.copy(self.state)

    def get_representation(self, state = None, turn = None):
        '''
        Returns state representation given a specified state and turn.

        Note:
        This function exists so that no hard coding for the connect-four
        environment has to exist in the Agent class.
        '''

        if state is None:
            state = self.state

        if turn is None:
            turn = self.turn

        representation = np.zeros(shape = (42, 2))
        p1 = np.where(state.flatten() == 1, 1, 0)
        p2 = np.where(state.flatten() == -1, 1, 0)

        representation[:, 0] = p1
        representation[:, 1] = p2

        return representation.reshape(1, -1)

    def show_board(self):
        '''
        Returns flipped string of board for human viewing.
        '''

        state_str = np.flip(np.where(self.state == 0, '-', np.where(self.state == 1, 'x', 'o')), axis = 0)
        return state_str

    def new_game(self):
        '''
        Initializes a new game.
        '''

        self.state = np.zeros(shape = (6, 7))
        self.turn = True
        return

    def move(self, actions):
        '''
        Alters board state and updates turn.

        Note:
        This alters the object, use view_move if you wish to see what a move's effects
        on the board and player turn would look like.
        '''

        actions = actions.flatten()
        
        pieces = np.abs(self.state).sum(axis = 0)
        legal = self.legal_moves()
        actions *= legal

        pos = np.unravel_index(np.argmax(actions, axis = None), actions.shape)[0]

        if self.turn:
            self.state[int(pieces[pos]), int(pos)] = 1

        else:
            self.state[int(pieces[pos]), int(pos)] = -1

        if self.turn:
            self.turn = False

        else:
            self.turn = True

        return

    def view_move(self, actions, turn = None, state = None):
        '''
        Returns future state and turn given current state, actions, and turn.

        Notes:
        This functions does not alter the board class, thus it provides itself as a useful
        helper function in the MCTS.
        '''

        if state is None:
            state = np.copy(self.state)

        if turn is None:
            turn = self.turn

        _state = np.copy(state)

        actions = actions.flatten()
        
        pieces = np.abs(_state).sum(axis = 0)
        legal = self.legal_moves(state = _state)
        actions *= legal

        pos = np.unravel_index(np.argmax(actions, axis = None), actions.shape)[0]

        if turn:
            _state[int(pieces[pos]), int(pos)] = 1

        else:
            _state[int(pieces[pos]), int(pos)] = -1

        if turn:
            turn = False

        else:
            turn = True

        return _state, turn

    def legal_moves(self, state = None):
        '''
        Returns vector of legal moves, legal moves being denoted as 1.
        '''

        if state is None:
            state = self.state

        pieces = np.abs(state).sum(axis = 0)
        legal = np.where(pieces < 6, 1, 0)

        return legal

    def check(self, state = None):
        '''
        Checks for winner in current board state.

        Notes:
        In the future, edit code to use np.trace for diagonal board checking.
        '''

        winner = None

        if state is None:
            state = self.state

        for i in range(4):
            
            value = state[:, i:i + 4].sum(axis = 1)

            if len(value[value == 4]) > 0:
                winner = True
                break

            if len(value[value == -4]) > 0:
                winner = False
                break

        for i in range(3):

            value = state[i:i + 4, :].sum(axis = 0)

            if len(value[value == 4]) > 0:
                winner = True
                break

            if len(value[value == -4]) > 0:
                winner = False
                break

        flip = np.flip(state, axis = 0)
        for y in range(flip.shape[1] - 3):
            for x in range(flip.shape[0] - 3):

                value = np.trace(flip[y:y+4], offset = x)

                if value == 4:
                    winner = True
                    break

                if value == -4:
                    winner = False
                    break

        flipped = np.flip(flip, axis = -1)
        for y in range(flipped.shape[1] - 3):
            for x in range(flipped.shape[0] - 3):

                value = np.trace(flipped[y:y+4], offset = x)

                if value == 4:
                    winner = True
                    break

                if value == -4:
                    winner = False
                    break

        if np.abs(state).sum() == 42:
            winner = 0

        return winner
