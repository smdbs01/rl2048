import argparse
import time
from pathlib import Path

import numba
import numpy as np

from ai.experience_collect import BestActionExperienceCollector
from ai.ntuple_utils import TUPLES
from env.bitboard import BitBoard
from main import load_agent

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
    "--load-file-name",
    "-lf",
    type=str,
    default="",
    help="(Only when load is true) file name of the agent to load. Leaving it empty will load the latest agent",
)
parser.add_argument(
    "--interval",
    "-i",
    type=int,
    default=200,
    help="Interval to render the trajectory, in milliseconds",
)

args = parser.parse_args()

SEED = args.seed
TUPLES_NAME = args.tuples
LOAD_FILE_NAME = args.load_file_name

MODEL_DIR = Path("models")

if __name__ == "__main__":
    np.random.seed(SEED)
    numba.set_num_threads(1)

    env = BitBoard()
    n_games, agent = load_agent(True, LOAD_FILE_NAME, TUPLES_NAME, MODEL_DIR)
    if not agent:
        print("No agent found. Exiting.")
        exit(1)

    print(f"Loaded agent with {n_games} games played.")
    experience_collector = BestActionExperienceCollector(env=env, ntuple_network=agent)

    start_time = time.time()

    trajectories = [
        experience_collector.collect_test(display=False) for _ in range(100)
    ]
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Average score:", np.mean([t.score for t in trajectories]))
    print("Average max tile:", np.mean([t.max_tile for t in trajectories]))
    print("Win rate:", np.mean([1 if t.max_tile >= 11 else 0 for t in trajectories]))
    print(
        "Average trajectory length:",
        np.mean([len(t.transitions) for t in trajectories]),
    )
