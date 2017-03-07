"""Game of Tic-Tac-Toe."""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

def _make_random_policy(np_random):
    def random_policy(state):
        possible_moves = TicTacToeEnv.get_possible_actions(state)
        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy


class TicTacToeEnv(gym.Env):

    """TicTacToe Environment. Play against a fixed opponent."""

    CROSS = 1
    CIRCLE = 2
    metadata = {"render.modes": ["ansi","human"]}

    def __init__(self, player_symbol, opponent, observation_type, illegal_move_mode, seed=None):
        """
        Args:
            player_symbol: The symbol of the agent (X or O)
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
            seed: Provide a random seed for the initial state
        """
        symbolmap = {
            'X': TicTacToeEnv.CROSS,
            'O': TicTacToeEnv.CIRCLE,
        }
        try:
            self.player_symbol = symbolmap[player_symbol]
        except KeyError:
            raise error.Error("player_symbol must be 'X' or 'O', not {}".format(player_symbol))

        self.board_size = 3

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign
        self.action_space = spaces.Discrete(self.board_size ** 2 + 1)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))
        self.done = False
        self.state = None
        self.to_play = None
        self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = _make_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
        self.state = np.zeros((self.board_size, self.board_size))
        self.state[:, :] = 0.0
        self.to_play = TicTacToeEnv.CROSS
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_symbol != self.to_play:
            a = self.opponent_policy(self.state)
            TicTacToeEnv.make_move(self.state, a, TicTacToeEnv.CROSS)
            self.to_play = TicTacToeEnv.CIRCLE
        return self.state

    def _step(self, action):
        assert self.to_play == self.player_symbol
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        if TicTacToeEnv.resign_move(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not TicTacToeEnv.valid_move(self.state, action):
            if self.illegal_move_mode == 'raise':
                raise
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))
        else:
            TicTacToeEnv.make_move(self.state, action, self.player_symbol)

        # Opponent play
        a = self.opponent_policy(self.state)

        # Move if there are moves left
        if a is not None:
            if TicTacToeEnv.resign_move(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            else:
                TicTacToeEnv.make_move(self.state, a, 1 - self.player_symbol)

        reward = TicTacToeEnv.game_finished(self.state)
        if self.player_symbol == TicTacToeEnv.CROSS:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # header
        outfile.write(' ' * 10)
        for j in range(board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '  | ')
        outfile.write('\n')
        # separator
        outfile.write(' ' * 9)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' * 5 +  str(i + 1) + '  |')
            for j in range(board.shape[1]):
                if board[i, j] == 0:
                    outfile.write('  _  ')
                elif board[i, j] == 1:
                    outfile.write('  X  ')
                else:
                    outfile.write('  O  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' * 9)
            outfile.write('-' * (board.shape[1] * 6 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_move(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def valid_move(board, action):
        coords = TicTacToeEnv.action_to_coordinate(board, action)
        if board[coords[0], coords[1]] == 0:
            return True
        else:
            return False

    @staticmethod
    def make_move(board, action, player):
        coords = TicTacToeEnv.action_to_coordinate(board, action)
        board[coords[0], coords[1]] = player

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def get_possible_actions(board):
        free_x, free_y = np.where(board[:, :] == 0)
        return [TicTacToeEnv.coordinate_to_action(board, [x, y]) for x, y in zip(free_x, free_y)]

    @staticmethod
    def game_finished(board):
        """
        Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
        """
        assert len(np.where(board[:, :] == 0)) > 0, "game not finished yet"

        # check rows
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2]:
                return int(board[i, 0])

        # check columns
        for j in range(3):
            if board[0, j] == board[1, j] == board[2, j]:
                return int(board[0, j])

        # check diagonals
        if (board[0, 0] == board[1, 1] == board[2, 2] or
            board[2, 0] == board[1, 1] == board[0, 2]):
           return int(board[1, 1])

        return 0
