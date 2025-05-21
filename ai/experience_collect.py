from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import sleep
from typing import Optional

import numpy as np
from numba import njit

from ai.ntuplenetwork import NTupleNetwork
from env.env import Action, Env


@dataclass
class Transition:
    """
    A class to represent a transition in the game.

    Attributes
    ----------
    state : np.uint64
        The state of the game, length 16 (4x4 board)
    action : Action
        The action taken in the game
    reward : float
        The reward received from the action
    after_state: np.uint64
        The state of the game after the action, length 16 (4x4 board)
    next_state: np.uint64
        The next state of the game, length 16 (4x4 board)
    score : int
        The total score obtained in the trajectory after this transition
    done : bool
        Whether the game is done or not
    """

    state: np.uint64
    action: Action
    reward: float
    after_state: np.uint64
    next_state: np.uint64
    score: int
    done: bool


@dataclass
class Trajectory:
    """
    A class to represent a trajectory in the game.

    Attributes
    ----------
    max_tile : int
        The maximum tile value in the trajectory
    score : int
        The total score of the trajectory
    transitions : list[Transition]
        The list of transitions in the trajectory
    """

    max_tile: int
    score: int
    transitions: list[Transition]


class ExperienceCollector(ABC):
    """
    Abstract base class for experience collectors.

    Attributes
    ----------
    env : Env
        The environment to collect experiences from
    ntuple_network : NTupleNetwork
        The n-tuple network to use for experience collection
    """

    def __init__(self, env: Env, ntuple_network: NTupleNetwork) -> None:
        self.env = env
        self.agent = ntuple_network

    def collect(self) -> Trajectory:
        """
        Collect a trajectory from the environment.

        Returns
        -------
        Trajectory
            The trajectory of the game
        """
        state = self._reset_env()
        transitions = []
        max_tile = 0
        score = 0
        while True:
            action = self._select_action(state)
            after_state, next_state, reward, done = self._step_env(action)

            max_tile = max(max_tile, self.env.get_max_tile())
            score += reward

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                after_state=after_state,
                next_state=next_state,
                score=score,
                done=done,
            )
            transitions.append(transition)

            if done:
                break

            state = next_state

        return Trajectory(max_tile=max_tile, score=score, transitions=transitions)

    def collect_test(self, display: bool = False, interval: int = 500) -> Trajectory:
        """
        Sample and render a trajectory.

        Parameters
        ----------
        display : bool
            Whether to display the game state

        interval : int
            The interval to render the game state. Only used if display is True.

        Returns
        -------
        Trajectory
            The trajectory of the game
        """
        state = self._reset_env()
        transitions = []
        max_tile = 0
        score = 0
        n_step = 0

        if display:
            print("-" * 20)
            print("Initial state:")
            print(self.env)
        while True:
            action = self._select_action(state)
            after_state, next_state, reward, done = self._step_env(action)

            max_tile = max(max_tile, self.env.get_max_tile())
            score += reward

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                after_state=after_state,
                next_state=next_state,
                score=score,
                done=done,
            )
            transitions.append(transition)

            if display:
                print("-" * 20)
                print(
                    f"Step {n_step} | Action: {action} | Reward: {reward} | Score: {score}"
                )
                print(self.env)

            n_step += 1

            if done:
                break
            state = next_state

            if display:
                sleep(interval / 1000)
        if display:
            print("-" * 20)
            print("Game over!")
            print(f"Final score: {score}")
            print(f"Max tile: {max_tile}")
            print(self.env)
        return Trajectory(max_tile=max_tile, score=score, transitions=transitions)

    @abstractmethod
    def _reset_env(self) -> np.uint64:
        """
        Reset the environment and return the initial state.

        Returns
        -------
        np.uint64
            The initial state of the environment
        """
        pass

    @abstractmethod
    def _select_action(self, state: np.uint64) -> Action:
        """
        Select an action based on the current state.

        Parameters
        ----------
        state : np.uint64
            The current state of the environment

        Returns
        -------
        Action
            The selected action
        """
        pass

    @abstractmethod
    def _step_env(self, action: Action) -> tuple[np.uint64, np.uint64, int, bool]:
        """
        Step the environment with the selected action.

        Parameters
        ----------
        action : Action
            The action to take in the environment

        Returns
        -------
        np.uint64
            The after state after taking the action

        np.uint64
            The next state of the environment

        int
            The reward received from the action

        bool
            Whether the game is done or not
        """
        pass


class BestActionExperienceCollector(ExperienceCollector):
    """
    Experience collector that collects experiences based on the best action.

    Attributes
    ----------
    env : Env
        The environment to collect experiences from
    ntuple_network : NTupleNetwork
        The n-tuple network to use for experience collection
    """

    def __init__(self, env: Env, ntuple_network: NTupleNetwork) -> None:
        super().__init__(env, ntuple_network)

    def _reset_env(self) -> np.uint64:
        """
        Reset the environment and return the initial state.

        Returns
        -------
        np.uint64
            The initial state of the environment
        """
        self.env.reset(None)
        return self.env.get_bb()

    def _select_action(self, state: np.uint64) -> Action:
        """
        Select an action based on the current state.

        Parameters
        ----------
        state : np.uint64
            The current state of the environment

        Returns
        -------
        Action
            The selected action
        """
        a, _, _, _ = self.agent._get_best_action(state)
        return a

    def _step_env(self, action: Action) -> tuple[np.uint64, np.uint64, int, bool]:
        """
        Step the environment with the selected action.

        Parameters
        ----------
        action : Action
            The action to take in the environment

        Returns
        -------
        np.uint64
            The after state after taking the action

        np.uint64
            The next state of the environment

        int
            The reward received from the action

        bool
            Whether the game is done or not
        """

        try:
            reward, changed, after_state = self.env.simulate(self.env.get_bb(), action)
        except ValueError:
            raise ValueError("Invalid action type")

        self.env.set_bb(after_state)

        if changed:
            self.env.add_random_tile()

        next_state = self.env.get_bb()

        self.env.check_game_over()
        done = self.env.is_game_over()

        return after_state, next_state, reward, done


class BestActionExperienceCollector_SS_TD(BestActionExperienceCollector):
    """
    Experience collector that collects experiences based on the best action,
    but add the ability to skip the start of the game and tile downgrading.

    Attributes
    ----------
    env : Env
        The environment to collect experiences from
    ntuple_network : NTupleNetwork
        The n-tuple network to use for experience collection
    """

    def __init__(self, env: Env, ntuple_network: NTupleNetwork) -> None:
        super().__init__(env, ntuple_network)

        self.last_trajectory: Optional[Trajectory] = None
        self.is_downgrade = False
        self.is_skip_start = False
        self.score = 0
        self.last_trajectory_length = 0

    def collect(self) -> Trajectory:
        state = self._reset_env()
        transitions = []
        max_tile = 0
        score = self.score
        while True:
            action = self._select_action(state)
            after_state, next_state, reward, done = self._step_env(action)

            score += reward
            max_tile = max(max_tile, self.env.get_max_tile())

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                after_state=after_state,
                next_state=next_state,
                score=score,
                done=done,
            )
            transitions.append(transition)

            if done:
                break

            state = next_state

            # if max_tile >= 12:
            #     self.is_downgrade = True

        trajectory = Trajectory(max_tile=max_tile, score=score, transitions=transitions)
        if score > 60000 and len(transitions) > 100:
            self.is_skip_start = True
        self.last_trajectory = trajectory  # Store the last trajectory
        self.is_downgrade = False
        self.score = 0
        return trajectory

    def _reset_env(self) -> np.uint64:
        """
        Reset the environment and return the initial state.

        Returns
        -------
        np.uint64
            The initial state of the environment
        """
        if self.is_skip_start and self.last_trajectory is not None:
            # Start from the middle of the last trajectory
            idx = len(self.last_trajectory.transitions) // 2
            initial_bb = self.last_trajectory.transitions[idx].next_state
            self.score = self.last_trajectory.transitions[idx].score
            self.env.reset(initial_bb)

            self.last_trajectory_length = len(self.last_trajectory.transitions) - idx
        else:
            self.env.reset(None)
            self.score = 0
            self.last_trajectory_length = 0

        self.is_skip_start = False
        return self.env.get_bb()

    def _select_action(self, state: np.uint64) -> Action:
        """
        Select an action based on the current state.

        Parameters
        ----------
        state : np.uint64
            The current state of the environment

        Returns
        -------
        Action
            The selected action
        """
        # tile downgrade
        bb = state
        if self.is_downgrade:
            bb = self._downgrade_tiles_jit(state)
        a, _, _, _ = self.agent._get_best_action(bb)
        return a

    @staticmethod
    @njit
    def _downgrade_tiles_jit(bb: np.uint64) -> np.uint64:
        tiles = np.empty(16, dtype=np.int32)
        for i in range(16):
            tiles[i] = int((bb >> np.uint64(4 * i)) & np.uint64(0xF))

        # find largest tile
        max_tile = 0
        for i in range(16):
            if tiles[i] > max_tile:
                max_tile = tiles[i]

        # find smallest missing tile
        present = np.zeros(max_tile + 2, dtype=np.bool_)
        for t in tiles:
            if t > 0:
                present[t] = True

        m = 1
        while m < present.size and present[m]:
            m += 1

        for i in range(16):
            if tiles[i] > m:
                tiles[i] -= 1

        new_bb = np.uint64(0)

        for i in range(16):
            new_bb |= np.uint64(tiles[i]) << np.uint64(4 * i)
        return new_bb
