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
        self.m = len(tuples)
        self.n = max([len(t) for t in tuples])
        self.tuples = np.full((self.m, self.n), -1, dtype=np.int32)
        self.lengths = np.array([len(t) for t in tuples], dtype=np.int32)

        for i, t in enumerate(tuples):
            self.tuples[i, : len(t)] = t

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
        self._v_from_bb_jit = self._build_jit_function()
        _ = self._v_from_bb_jit(np.uint64(0))

    def _build_jit_function(self):
        tuples = self.tuples
        lut = self.lut
        m = self.m
        lengths = self.lengths
        TILE_BITS = self.TILE_BITS
        TILE_MASK = self.TILE_MASK

        @njit(parallel=True)
        def _v_jit(bb: np.uint64) -> float:
            v = 0.0
            for i in prange(m):
                idx = 0
                n = lengths[i]
                for j in range(n):
                    pos = tuples[i, j]
                    shift = pos * TILE_BITS
                    tile = (bb >> shift) & TILE_MASK
                    idx += tile * (1 << (j * TILE_BITS))
                w = lut[i, idx]
                v += w
            return v

        return _v_jit

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
        res = self._v_from_bb_jit(bb)
        return res

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

    def calculate_td_error(self, s_after: np.uint64, s_next: np.uint64) -> float:
        """
        Calculate the TD error for the given states.

        Delta = r_{t+1} + V(s'_{t+1}) - V(s'_{t})

        Parameters
        ----------
        s_after : np.uint64
            The after state (board after the action but before a new tile is added)

        s_next : np.uint64
            The next state (board after the action and a new tile is added)

        Returns
        -------
        float
            The TD error
        """

        _, _, r_next, v_next_after = self._get_best_action(s_next)
        v_after = self.V(s_after)

        return r_next + v_next_after - v_after

    @staticmethod
    @njit
    def _update_weights_jit(
        lut: np.ndarray,
        tuples: np.ndarray,
        lengths: np.ndarray,
        s_after: np.uint64,
        delta: float,
        alpha: float,
        E: np.ndarray,
        A: np.ndarray,
        TILE_BITS: int,
        TILE_MASK: np.uint64,
        is_tc: bool,
    ):
        m, _ = tuples.shape
        for i in prange(m):
            idx = 0
            n = lengths[i]
            for j in range(n):
                pos = tuples[i, j]
                shift = pos * TILE_BITS
                tile = (s_after >> shift) & TILE_MASK
                idx += tile << (j * TILE_BITS)

            beta = 1.0
            if is_tc:
                beta = np.abs(E[i, idx]) / A[i, idx] if A[i, idx] != 0 else 1
                if beta > 10:
                    print(f"beta is too large: {beta}")
            lut[i, idx] += delta * beta * alpha / m
            if is_tc:
                E[i, idx] += delta
                A[i, idx] += np.abs(delta)

    def update_weights(
        self,
        s_after: np.uint64,
        cumulative_delta: float,
        alpha: float = 0.1,
        is_tc: bool = False,
    ):
        if not np.isfinite(cumulative_delta):
            print(f"cumulative_delta is not finite: {cumulative_delta}")
            return
        # Update the weights
        NTupleNetwork._update_weights_jit(
            self.lut,
            self.tuples,
            self.lengths,
            s_after,
            cumulative_delta,
            alpha,
            self.E,
            self.A,
            self.TILE_BITS,
            self.TILE_MASK,
            is_tc,
        )
