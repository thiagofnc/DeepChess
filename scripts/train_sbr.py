"""Fit a constant on training games and evaluate on held-out games."""

import argparse
import json
from pathlib import Path
from statistics import mean

from chess_data import DEFAULT_INPUT, load_games, numeric_scores, regression_metrics, split_games


def fit_and_evaluate(path=DEFAULT_INPUT, seed=42, max_games=None):
    splits = split_games(load_games(path, max_games), seed=seed)
    scores = {name: numeric_scores(games) for name, games in splits.items()}
    for name, values in scores.items():
        if not values:
            raise ValueError(f"{name} split contains no numeric evaluations")
    average_eval = mean(scores["train"])
    return {
        "seed": seed,
        "mean_eval_pawns": average_eval,
        "game_counts": {name: len(games) for name, games in splits.items()},
        "position_counts": {name: len(values) for name, values in scores.items()},
        "metrics": {
            name: regression_metrics(values, [average_eval] * len(values))
            for name, values in scores.items() if name != "train"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-games", type=int, help="Optional debugging limit (default: all)")
    args = parser.parse_args()
    print(json.dumps(fit_and_evaluate(args.input, args.seed, args.max_games), indent=2))


if __name__ == "__main__":
    main()
