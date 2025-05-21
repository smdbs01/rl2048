from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Env(ABC):
    """
    Abstract base class for the 2048 game environment.
    """

    def __init__(self, initial_board: Optional[np.uint64] = None) -> None:
        """
        Initialize the environment with the given initial board.

        Parameters
        ----------
        initial_board : Optional[np.uint64]
            The initial state of the board. If None, a new board is created.
        """
        self.reset(initial_board)

    def reset(self, board: Optional[np.uint64] = None) -> None:
        """
        Resets the environment to the initial state.

        Parameters
        ----------
        board : Optional[np.uint64]
            The initial state of the board. If None, a new board is created.
        """
        self.clear()
        if board is not None:
            self.set_bb(board)
        else:
            self.add_random_tile()
            self.add_random_tile()

        self.check_game_over()

    @abstractmethod
    def clear(self) -> None:
        """
        Clears the board
        """
        pass

    @abstractmethod
    def check_game_over(self) -> None:
        """
        Checks if the game is over.
        The game is over if there are no empty spaces and no possible merges.
        """
        pass

    @abstractmethod
    def add_random_tile(self) -> None:
        """
        Adds a random tile (2 or 4) to an empty space on the board.
        """
        pass

    @abstractmethod
    def simulate(self, bb: np.uint64, action: Action) -> tuple[int, bool, np.uint64]:
        """
        Simulates the specified action and returns the score gained and merged board from the move.
        Does not change the internal state of this Board object.

        Parameters
        ----------
        bb : np.uint64
            The current state of the board.

        action : Action
            The action to perform (UP, DOWN, LEFT, RIGHT).

        Returns
        -------
        int
            The score gained from the move.

        bool
            Whether the board changed after the move.

        np.uint64
            The new board after the move.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Returns whether the game is over.
        """
        pass

    @abstractmethod
    def get_max_tile(self) -> int:
        """
        Returns the maximum tile value on the board.
        """
        pass

    @abstractmethod
    def get_bb(self) -> np.uint64:
        """
        Returns the current state of the board as a 64-bit integer.
        """
        pass

    @abstractmethod
    def set_bb(self, board: np.uint64) -> None:
        """
        Sets the current state of the board from a 64-bit integer.
        """
        pass

    @abstractmethod
    def get_tile(self, index: int) -> int:
        """
        Returns the value of the tile at the given index.
        """
        pass

    @abstractmethod
    def set_tile(self, index: int, value: int) -> None:
        """
        Sets the value of the tile at the given index.
        """
        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the board.
        """
        board_str = "\n" + "-" * 20 + "\n"
        board = self.get_bb()
        for i in range(4):
            for j in range(4):
                tile = (board >> (i * 16 + j * 4)) & 0xF
                board_str += f"{tile:4d}"
            board_str += "\n" + "-" * 20 + "\n"
        return board_str

    def __repr__(self) -> str:
        """
        Returns a string representation of the board.
        """
        return self.__str__()
