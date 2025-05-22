from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import sleep

import numpy as np

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
    delta: float
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


class Trainer(ABC):
    """
    Abstract class for a trainer.

    Attributes
    ----------
    env : Env
        The environment to collect experiences from
    ntuple_network : NTupleNetwork
        The n-tuple network to use for experience collection
    alpha : float
        The learning rate for the agent
    """

    def __init__(
        self, env: Env, ntuple_network: NTupleNetwork, alpha: float = 0.1
    ) -> None:
        self.env = env
        self.agent = ntuple_network
        self.alpha = alpha

    def set_alpha(self, alpha: float) -> None:
        """
        Set the learning rate for the agent.

        Parameters
        ----------
        alpha : float
            The learning rate for the agent
        """
        self.alpha = alpha

    def collect(self) -> Trajectory:
        """
        Collect a trajectory from the environment.
        Update the weights of the agent based on the collected trajectory.

        Returns
        -------
        Trajectory
            The trajectory of the game
        """
        state = self._reset_env()
        transitions: list[Transition] = []
        max_tile = 0
        score = 0
        while True:
            action = self._select_action(state)
            after_state, next_state, reward, done = self._step_env(action)

            max_tile = max(max_tile, self.env.get_max_tile())
            score += reward

            delta = self.agent.calculate_td_error(after_state, next_state)

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                after_state=after_state,
                next_state=next_state,
                delta=delta,
                score=score,
                done=done,
            )

            if done:
                break

            transitions.append(transition)  # don't append the failing transition

            state = next_state

        self.update_weights(transitions)

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
                delta=0,
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

    @abstractmethod
    def update_weights(self, transitions: list[Transition]) -> None:
        """
        Update the weights of the agent based on the collected transitions.

        Parameters
        ----------
        transitions : list[Transition]
            The list of transitions to update the weights with
        """
        pass


class BestActionTDTrainer(Trainer):
    """
    Trainer that collects experiences based on the best action.
    This class updates the weights of the agent based on the collected experiences and calculates the delta using N-step TD.

    Attributes
    ----------
    env : Env
        The environment to collect experiences from
    ntuple_network : NTupleNetwork
        The n-tuple network to use for experience collection
    alpha : float
        The learning rate for the agent
    td_h : int
        The number of steps to look ahead for TD calculation
    td_lambda : float
        The decay factor for the TD error
    """

    def __init__(
        self,
        env: Env,
        ntuple_network: NTupleNetwork,
        alpha: float = 0.1,
        td_h: int = 3,
        td_lambda: float = 0.5,
    ) -> None:
        super().__init__(env, ntuple_network, alpha)

        self.td_h = td_h
        self.td_lambda = td_lambda
        self.is_tc = False

    def set_tc(self, is_tc: bool) -> None:
        """
        Set the Temporal Coherence training flag.

        Parameters
        ----------
        is_tc : bool
            Whether to use Temporal Coherence training or not
        """
        self.is_tc = is_tc

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

    def update_weights(self, transitions: list[Transition]) -> None:
        """
        Update the weights of the agent based on the collected transitions.

        Parameters
        ----------
        transitions : list[Transition]
            The list of transitions to update the weights with
        """

        buffer = []

        for idx, tr in enumerate(transitions):
            buffer.append((tr.after_state, tr.delta))

            if idx >= self.td_h:
                cumulative_delta = 0
                for i in range(self.td_h + 1):
                    if (idx - self.td_h + i) < len(buffer):
                        delta = buffer[idx - self.td_h + i][1]
                        cumulative_delta += self.td_lambda**i * delta

                oldest_after_state = buffer[idx - self.td_h][0]
                self.agent.update_weights(
                    s_after=oldest_after_state,
                    cumulative_delta=cumulative_delta,
                    alpha=self.alpha,
                    is_tc=self.is_tc,
                )

        # Update the weights for the last h-1 transitions
        for k in range(max(0, len(buffer) - self.td_h), len(buffer)):
            cumulative_delta = 0
            for i in range(self.td_h + 1):
                if k + i < len(buffer):
                    delta = buffer[k + i][1]
                    cumulative_delta += self.td_lambda**i * delta

            self.agent.update_weights(
                s_after=buffer[k][0],
                cumulative_delta=cumulative_delta,
                alpha=self.alpha,
                is_tc=self.is_tc,
            )
