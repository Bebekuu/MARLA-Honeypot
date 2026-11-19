"""Run the pilot experiment for several random networks (seeds) and save one JSON file per seed.

Example:
    python scripts/run_experiment.py --seeds 20 --workers 2
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from marla_honeypot import run_seed


def _run(args):
    seed, episodes, out_dir = args
    result = run_seed(seed, train_eps=episodes)
    path = Path(out_dir) / f"seed{seed}.json"
    path.write_text(json.dumps(result))
    return seed, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20, help="number of random networks (seeds 0..N-1)")
    parser.add_argument("--episodes", type=int, default=40000, help="training episodes per learner")
    parser.add_argument("--workers", type=int, default=2, help="parallel processes")
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(s, args.episodes, args.out) for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for seed, path in pool.map(_run, jobs):
            print(f"seed {seed:2d} -> {path}")


if __name__ == "__main__":
    main()
