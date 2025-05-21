import argparse
import gc
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numba
import numpy as np
from tqdm import tqdm

from ai.experience_collect import BestActionExperienceCollector
from ai.ntuple_utils import TUPLES, get_symmetric_tuples
from ai.ntuplenetwork import NTupleNetwork
from env.bitboard import BitBoard

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-ss", type=int, default=42, help="Random seed")
parser.add_argument(
    "--tuples",
    "-t",
    type=str,
    default="Matsuzaki",
    help=f"N-tuple network name. One of {list(TUPLES.keys())}",
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=0.1,
    help="Learning rate for the agent",
)
parser.add_argument(
    "--tc-ratio",
    "-tc",
    type=float,
    default=0.1,
    help="Ratio of Temporal Coherence training to the total training",
)
parser.add_argument(
    "--n-iterations",
    "-ni",
    type=int,
    default=5000,
    help="Number of iterations to train the agent",
)
parser.add_argument(
    "--n-episodes",
    "-ne",
    type=int,
    default=100,
    help="Number of episodes to train the agent in each iteration",
)
parser.add_argument(
    "--save-interval",
    "-si",
    type=int,
    default=100,
    help="Interval to save the agent",
)
parser.add_argument(
    "--load",
    action=argparse.BooleanOptionalAction,
    help="Whether to load an existing agent",
    default=False,
)
parser.add_argument(
    "--load-file-name",
    "-lf",
    type=str,
    default="",
    help="(Only when load is true) file name of the agent to load. Leaving it empty will load the latest agent",
)
parser.add_argument(
    "--output-name",
    "-o",
    type=str,
    default="",
    help="File name of the agent to save. If empty, the default name will be used",
)
parser.add_argument(
    "--tqdm",
    action=argparse.BooleanOptionalAction,
    help="Whether to show tqdm progress bar",
    default=True,
)

args = parser.parse_args()

SEED: int = args.seed
TUPLES_NAME: str = args.tuples
LEARNING_RATE: float = args.learning_rate
TC_RATIO: float = args.tc_ratio
N_ITERATIONS: int = args.n_iterations
N_EPISODES: int = args.n_episodes
SAVE_INTERVAL: int = args.save_interval
LOAD: bool = args.load
LOAD_FILE_NAME: str = args.load_file_name
OUTPUT_NAME: str = args.output_name
TQDM: bool = args.tqdm

MODEL_DIR = Path("models")
HISTORY_DIR = Path("history")
HISTORY_FIELD = ["max_tiles", "scores"]


def load_agent(
    is_load: bool,
    load_file_name: str,
    tuples_name: str,
    model_dir: Path,
) -> tuple[int, Optional[NTupleNetwork]]:
    """
    Load an existing agent from a file.

    Parameters
    ----------
    is_load : bool
        Whether to load an existing agent.

    load_file_name : str
        The file name of the agent to load. If empty, the latest agent will be loaded.

    tuples_name : str
        The name of the N-tuple network to load.

    model_dir : Path
        The directory where the agent is saved.

    Returns
    -------
    tuple[int, NTupleNetwork]
        The number of games played, the agent
    """
    model_dir.mkdir(exist_ok=True)

    if is_load:
        if load_file_name:
            save = model_dir / load_file_name
            if not save.exists():
                print(f"File {save} does not exist. Creating a new agent.")
                return 0, None
            with open(save, "rb") as f:
                n_games, tuples_name, lut = pickle.load(f)
                agent = NTupleNetwork(tuples=get_symmetric_tuples(tuples_name), lut=lut)
                return n_games, agent
        else:
            print("Loading the latest agent...")
            saves = list(model_dir.glob("*.pkl"))
            if not saves:
                print("No saved agents found. Creating a new agent.")
                return 0, None
            save = max(
                [s for s in saves if tuples_name in s.name], key=os.path.getctime
            )
            print(f"Loading {save}...")
            with open(save, "rb") as f:
                n_games, tuples_name, lut = pickle.load(f)
                agent = NTupleNetwork(tuples=get_symmetric_tuples(tuples_name), lut=lut)
                return n_games, agent
    print("Loading is disabled. Creating a new agent.")
    return 0, None


def save_agent(
    agent: NTupleNetwork, n_games: int, history: dict, tuples_name: str
) -> None:
    """
    Save the agent to a file.

    Parameters
    ----------
    agent : NTupleNetwork
        The agent to save.
    n_games : int
        The number of games played by the agent.
    history : dict
        The history of scores and max tiles.
    """

    name = OUTPUT_NAME if OUTPUT_NAME else f"{agent.__class__.__name__}"
    model_filename = MODEL_DIR / f"{name}_{tuples_name}_{n_games}games_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump((n_games, tuples_name, agent.lut), f)
    print(f"Agent saved to {model_filename}")

    history_filename = HISTORY_DIR / f"{name}_{tuples_name}_{n_games}games_history.pkl"
    with open(history_filename, "wb") as f:
        pickle.dump(history, f)
    print(f"History saved to {history_filename}")


if __name__ == "__main__":
    np.random.seed(SEED)
    numba.set_num_threads(1)
    gc.disable()

    env = BitBoard()
    n_games, agent = load_agent(LOAD, LOAD_FILE_NAME, TUPLES_NAME, MODEL_DIR)
    history = {k: [] for k in HISTORY_FIELD}
    if agent is None:
        tuples = get_symmetric_tuples(TUPLES_NAME)
        agent = NTupleNetwork(tuples, initial_weight=10_000)
        history = {k: [] for k in HISTORY_FIELD}
        n_games = 0

    experience_collector = BestActionExperienceCollector(env, agent)

    print("Starting training...")
    alpha = 0.1
    is_tc = False
    try:
        for update in range(n_games // N_EPISODES, N_ITERATIONS):
            max_tiles = []
            scores = []

            if update == (N_ITERATIONS - TC_RATIO * N_ITERATIONS):
                is_tc = True
                alpha = 1
                print("Starting Temporal Coherence training")

            t = range(N_EPISODES)
            if TQDM:
                t = tqdm(t, desc=f"Iter {update}", leave=False)

            start_time = time.time()
            for episode in t:
                trajectory = experience_collector.collect()

                if TQDM and type(t) is tqdm:
                    t.set_postfix(
                        {
                            "max_tile": trajectory.max_tile,
                            "score": trajectory.score,
                            "alpha": alpha,
                        }
                    )
                for tr in trajectory.transitions[:-1][::-1]:
                    agent.update_weights(
                        tr.after_state, tr.next_state, alpha=alpha, is_tc=is_tc
                    )

                max_tiles.append(trajectory.max_tile)
                scores.append(trajectory.score)

                n_games += 1

            end_time = time.time()

            history["max_tiles"].append(np.mean(max_tiles))
            history["scores"].append(np.mean(scores))

            print(
                f"Iteration {update + 1}/{N_ITERATIONS}: "
                f"Max Tile: {np.mean(max_tiles):.2f} "
                f"Score: {np.mean(scores):.2f} "
                f"Time: {end_time - start_time:.2f}s "
            )

            if (update + 1) % SAVE_INTERVAL == 0:
                save_agent(agent, n_games, history, TUPLES_NAME)
                history = {k: [] for k in HISTORY_FIELD}  # reset history

            gc.collect()

        print("Training completed successfully.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the current model...")
    finally:
        # Save the agent at the end of training
        if input("\nSave the current model? (y/n) ").lower() == "y":
            save_agent(agent, n_games, history, TUPLES_NAME)
