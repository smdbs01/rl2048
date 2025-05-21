from typing import Optional

import numpy as np
from numpy import random

from env.env import Action, Env

""" 

Adopted from https://github.com/nicholasburden/2048

"""


class BitBoard(Env):
    ROW_MASK = np.uint64(0xFFFF)
    COL_MASK = np.uint64(0x000F000F000F000F)
    TILE_MASK = np.uint64(0xF)
    TILE_BITS = 4
    MERGE_TABLE_LEFT: dict[np.uint64, tuple[np.uint64, int]] = {}
    MERGE_TABLE_RIGHT: dict[np.uint64, tuple[np.uint64, int]] = {}

    def __init__(self, initial_board: Optional[np.uint64] = None) -> None:
        self.bitboard: np.uint64
        self.game_over = False

        self.reset(initial_board)
        self._init_merge_table()

    def _init_merge_table(self) -> None:
        """
        Initialize the merge table for the game.
        """
        if BitBoard.MERGE_TABLE_LEFT and BitBoard.MERGE_TABLE_RIGHT:
            return

        for x in range(1 << 16):  # For every row
            bb_row = np.uint64(x)
            row = self._to_row(bb_row)
            merged_line, score = self._merge_right(row)
            result = self._to_bitboard(merged_line)

            rev_row = self._reverse_bitrow(bb_row)
            rev_result = self._reverse_bitrow(result)

            BitBoard.MERGE_TABLE_RIGHT[bb_row] = (result, score)
            BitBoard.MERGE_TABLE_LEFT[rev_row] = (rev_result, score)

    def _to_row(self, bb: np.uint64) -> np.ndarray:
        """
        Convert a bitboard row to a 1D array.

        Example:
        0x2110 -> [0, 1, 1, 2]

        Parameters
        ----------
        bb : np.uint64
            The bitboard row to convert

        Returns
        -------
        np.ndarray
            The converted 1D array
        """
        row = np.zeros(4, dtype=np.uint8)
        for j in range(4):
            row[j] = int((bb >> (self.TILE_BITS * j)) & self.TILE_MASK)
        return row

    def _to_bitboard(self, row: np.ndarray) -> np.uint64:
        """
        Convert a 1D array to a bitboard row

        Parameters
        ----------
        row : np.ndarray
            The 1D array to convert

        Returns
        -------
        np.uint64
            The converted bitboard row
        """
        bb = np.uint64(0)
        for j, tile in enumerate(row):
            if tile:
                bb |= np.uint64(tile) << np.uint64(self.TILE_BITS * j)
        return bb

    def _reverse_bitrow(self, bb: np.uint64) -> np.uint64:
        """
        Reverse the bitboard row.

        Parameters
        ----------
        bb : np.uint64
            The bitboard row to reverse

        Returns
        -------
        np.uint64
            The reversed bitboard row
        """
        a1 = (bb & np.uint64(0xF000)) >> np.uint64(12)
        a2 = (bb & np.uint64(0x0F00)) >> np.uint64(4)
        a3 = (bb & np.uint64(0x00F0)) << np.uint64(4)
        a4 = (bb & np.uint64(0x000F)) << np.uint64(12)
        return a1 | a2 | a3 | a4

    def _merge_right(self, row: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Merge the tiles in a row to the right.

        Parameters
        ----------
        row : np.ndarray
            The row to merge

        Returns
        -------
        np.ndarray
            The merged row

        int
            The score gained from the merge
        """

        non_zero = row[row != 0][::-1]
        merged = []
        score = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] + 1)
                score += np.uint64(2) ** (non_zero[i] + 1)
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        new_line = np.zeros(4, dtype=np.uint8)
        if len(merged) > 0:
            new_line[-len(merged) :] = merged[::-1]
        return new_line, score

    def simulate(self, bb: np.uint64, action: Action) -> tuple[int, bool, np.uint64]:
        """
        Simulate the specified action and return the score gained and merged board from the move.
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

        Raises
        ------
        ValueError
            If the action is invalid.
        """
        if action not in Action:
            raise ValueError("Invalid action type")

        old = bb
        tmp = np.uint64(0)
        total_score = 0
        changed = False
        if action == Action.UP or action == Action.DOWN:
            trans = self._transpose(old)
            for i in range(4):
                row = (trans >> np.uint64(i * 16)) & self.ROW_MASK
                if action == Action.UP:
                    merged, s = BitBoard.MERGE_TABLE_LEFT[row]
                else:
                    merged, s = BitBoard.MERGE_TABLE_RIGHT[row]
                tmp |= merged << np.uint64(i * 16)
                total_score += int(s)
                if row != merged:
                    changed = True
            new_bb = self._transpose(tmp)
        else:
            for i in range(4):
                row = (old >> np.uint64(i * 16)) & self.ROW_MASK
                if action == Action.LEFT:
                    merged, s = BitBoard.MERGE_TABLE_LEFT[row]
                else:
                    merged, s = BitBoard.MERGE_TABLE_RIGHT[row]
                tmp |= merged << np.uint64(i * 16)
                total_score += int(s)
                if row != merged:
                    changed = True
            new_bb = tmp

        return total_score, changed, new_bb

    def _transpose(self, bb: np.uint64) -> np.uint64:
        """
        Transpose the bitboard.

        Parameters
        ----------
        bb : np.uint64
            The bitboard to transpose

        Returns
        -------
        np.uint64
            The transposed bitboard
        """
        a1 = bb & np.uint64(0xF0F00F0FF0F00F0F)
        a2 = bb & np.uint64(0x0000F0F00000F0F0)
        a3 = bb & np.uint64(0x0F0F00000F0F0000)
        a = a1 | (a2 << np.uint64(12)) | (a3 >> np.uint64(12))
        b1 = a & np.uint64(0xFF00FF0000FF00FF)
        b2 = a & np.uint64(0x00FF00FF00000000)
        b3 = a & np.uint64(0x00000000FF00FF00)
        return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))

    def clear(self) -> None:
        """
        Clear the bitboard.
        """
        self.bitboard = np.uint64(0)
        self.game_over = False

    def check_game_over(self) -> None:
        """
        Check if the game is over.
        """
        for i in range(16):
            if self.get_tile(i) == 0:
                self.game_over = False
                return
        for move in Action:
            _, changed, _ = self.simulate(self.bitboard, move)
            if changed:
                self.game_over = False
                return
        self.game_over = True

    def add_random_tile(self) -> None:
        """
        Add a random tile (2 or 4) to an empty space on the board.
        """
        indices = [i for i in range(16) if self.get_tile(i) == 0]
        if not indices:
            return
        idx = random.choice(indices)
        val = 1 if random.rand() < 0.9 else 2
        self.set_tile(idx, val)

    def is_game_over(self) -> bool:
        return self.game_over

    def get_max_tile(self) -> int:
        return max(self.get_tile(i) for i in range(16))

    def get_tile(self, index: int) -> int:
        if index < 0 or index >= 16:
            raise ValueError("Index must be between 0 and 15")
        return ((self.bitboard >> (index * self.TILE_BITS)) & self.TILE_MASK).item()

    def set_tile(self, index: int, value: int) -> None:
        if index < 0 or index >= 16:
            raise ValueError("Index must be between 0 and 15")
        if value < 0 or value > 15:
            raise ValueError("Value must be between 0 and 15")
        mask = self.TILE_MASK << np.uint64(index * self.TILE_BITS)
        self.bitboard &= ~mask
        self.bitboard |= np.uint64(value) << np.uint64(index * self.TILE_BITS)

    def get_bb(self) -> np.uint64:
        return self.bitboard

    def set_bb(self, board: np.uint64) -> None:
        self.bitboard = board
        self.check_game_over()
