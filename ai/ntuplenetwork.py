"""

Adopted from https://github.com/alanhyue/RL-2048-with-n-tuple-network.

"""

from typing import Optional

import numpy as np
from numba import njit, prange

from env.bitboard import BitBoard
from env.env import Action


class NTupleNetwork:
    """
    N-tuple network for the 2048 game.

    Use OTD-TC (Optimistic TD with Temporal Coherense) to update the weights.
    """

    TILE_BITS = 4
    TILE_MASK = np.uint64(0xF)

    def __init__(
        self,
        tuples: list[tuple[int]],
        initial_weight: float = 0.0,
        lut: Optional[np.ndarray] = None,
        tc_tables: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Initialize the NTupleNetwork with the given tuples and initial weight.

        Parameters
        ----------
        tuples : list[tuple[int]]
            A list of tuples representing the n-tuples.
        initial_weight : float, optional
            The initial weight for the n-tuples, by default 0
        """
        self.tuples = np.array(tuples, dtype=np.int32)
        self.m, self.n = self.tuples.shape

        if lut is None:
            self.lut = np.full((self.m, 16**self.n), initial_weight, dtype=np.float32)
        else:
            assert lut.shape == (self.m, 16**self.n)
            self.lut = lut

        if tc_tables is None:
            self.E = np.zeros((self.m, 16**self.n), dtype=np.float32)
            self.A = np.zeros((self.m, 16**self.n), dtype=np.float32)
        else:
            assert tc_tables[0].shape == (self.m, 16**self.n)
            assert tc_tables[1].shape == (self.m, 16**self.n)
            self.E = tc_tables[0]
            self.A = tc_tables[1]

        self._precompute_jit()

    def _precompute_jit(self) -> None:
        self._get_tuple_index_jit, self._v_from_bb_jit = self._build_jit_function()
        _ = self._get_tuple_index_jit(np.uint64(0), np.zeros(self.n, dtype=np.int32))
        _ = self._v_from_bb_jit(np.uint64(0))

    def _build_jit_function(self):
        tuples = self.tuples
        lut = self.lut
        n = self.n
        TILE_BITS = self.TILE_BITS
        TILE_MASK = self.TILE_MASK

        @njit(parallel=True)
        def _get_tuple_index_jit(bb: np.uint64, t: np.ndarray) -> int:
            index = 0
            for j in prange(n):
                pos = t[j]
                shift = pos * TILE_BITS
                tile = (bb >> shift) & TILE_MASK
                index += tile * (1 << (j * TILE_BITS))
            return index

        @njit(parallel=True)
        def _v_jit(bb: np.uint64) -> float:
            v = 0.0
            nt = tuples.shape[0]
            for i in prange(nt):
                idx = 0
                for j in prange(n):
                    pos = tuples[i, j]
                    shift = pos * TILE_BITS
                    tile = (bb >> shift) & TILE_MASK
                    idx += tile * (1 << (j * TILE_BITS))
                v += lut[i, idx]
            return v

        return _get_tuple_index_jit, _v_jit

    def V(self, bb: np.uint64) -> float:
        """
        Calculate the value of the given state

        Parameters
        ----------
        bb : np.uint64
            The state of the game, length 16 (4x4 board)

        Returns
        -------
        float
            The value of the state
        """
        return self._v_from_bb_jit(bb)

    def _get_best_action(self, bb: np.uint64) -> tuple[Action, np.uint64, int, float]:
        """
        Get the best action for the given state.

        Parameters
        ----------
        bb : np.uint64
            The state of the game, length 16 (4x4 board)

        Returns
        -------
        tuple[Action, np.uint64, int, float]
            The best action, the new state after the action, the score of the action,
            and the value of the new state
        """

        best_action = Action.UP
        best_value = -np.inf
        best_score = -1
        best_board_value = -np.inf
        best_bb = bb

        board = BitBoard()
        for action in Action:
            score, changed, new_bb = board.simulate(bb, action)
            if changed:
                value = self.V(new_bb)
                if value + score > best_value:
                    best_value = value + score
                    best_action = action
                    best_score = score
                    best_board_value = value
                    best_bb = new_bb

        return best_action, best_bb, best_score, best_board_value

    def _calculate_td_error(
        self, s_after: np.uint64, s_next: np.uint64, gamma: float = 1
    ) -> float:
        """
        Calculate the TD error for the given states and gamma.

        Delta = r_{t+1} + V(s'_{t+1}) - V(s'_{t})

        Parameters
        ----------
        s_after : np.uint64
            The after state (board after the action but before a new tile is added)

        s_next : np.uint64
            The next state (board after the action and a new tile is added)

        gamma : float
            The discount factor, by default 1

        Returns
        -------
        float
            The TD error
        """

        _, _, r_next, v_next_after = self._get_best_action(s_next)
        v_after = self.V(s_after)

        return r_next + gamma * v_next_after - v_after

    @staticmethod
    @njit(parallel=True)
    def _update_weights_jit(
        lut: np.ndarray,
        tuples: np.ndarray,
        s_after: np.uint64,
        s_next: np.uint64,
        delta: float,
        alpha: float,
        E: np.ndarray,
        A: np.ndarray,
        TILE_BITS: int,
        TILE_MASK: np.uint64,
        is_tc: bool,
    ):
        m, n = tuples.shape
        for i in prange(m):
            idx = np.uint64(0)
            for j in range(n):
                pos = tuples[i, j]
                shift = pos * TILE_BITS
                tile = (s_after >> shift) & TILE_MASK
                idx += tile << (j * TILE_BITS)

            beta = 1.0
            if is_tc:
                beta = np.abs(E[i, idx]) / A[i, idx] if A[i, idx] != 0 else 1
            lut[i, idx] += delta * beta * alpha / m
            if is_tc:
                E[i, idx] += delta
                A[i, idx] += np.abs(delta)

    def update_weights(
        self,
        s_after: np.uint64,
        s_next: np.uint64,
        alpha: float = 0.01,
        is_tc: bool = False,
    ):
        # Calculate the TD error
        delta = self._calculate_td_error(s_after, s_next)

        # Update the weights
        NTupleNetwork._update_weights_jit(
            self.lut,
            self.tuples,
            s_after,
            s_next,
            delta,
            alpha,
            self.E,
            self.A,
            self.TILE_BITS,
            self.TILE_MASK,
            is_tc,
        )
