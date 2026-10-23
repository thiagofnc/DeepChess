"""Stream Lichess games and save only games containing Stockfish evaluations."""

import argparse
from datetime import date, datetime, time
import json
from pathlib import Path
import re
import sys


DATASET = "Lichess/standard-chess-games"
# Accept pawn scores and mate scores, including optional analysis depth.
EVAL_PATTERN = re.compile(
    r"\[%eval\s+(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)|#[+-]?\d+)"
    r"(?:\s*,\s*\d+)?\s*\]"
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def json_default(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_evaluated_games(games, output, max_games, progress_every=10_000):
    """Save complete matching rows as JSONL; never overwrite an existing file."""
    scanned = saved = 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for game in games:
            scanned += 1
            if "movetext" not in game:
                raise ValueError("Dataset row has no 'movetext' field; check its schema.")
            movetext = game["movetext"]
            if isinstance(movetext, str) and EVAL_PATTERN.search(movetext):
                handle.write(json.dumps(game, default=json_default, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= max_games:
                    break
            if scanned % progress_every == 0:
                print(f"Scanned {scanned:,} games; saved {saved:,}.", flush=True)
    return scanned, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-games", type=positive_int, default=10_000,
                        help="Number of matching games to save (default: 10000).")
    parser.add_argument("--output", type=Path,
                        default=Path("data/stockfish_games.jsonl"))
    parser.add_argument("--year", type=positive_int,
                        help="Only scan this year's Parquet files.")
    parser.add_argument("--month", type=int, choices=range(1, 13),
                        help="Only scan this month (requires --year).")
    parser.add_argument("--revision", default="main",
                        help="Hugging Face revision or commit hash (default: main).")
    args = parser.parse_args()
    if args.month is not None and args.year is None:
        parser.error("--month requires --year")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new --output path.")

    try:
        from datasets import load_dataset
    except ImportError:
        parser.exit(1, "Install dependencies first: python -m pip install -r requirements.txt\n")

    options = {}
    if args.year is not None:
        month = f"{args.month:02d}" if args.month is not None else "*"
        options["data_files"] = f"data/year={args.year}/month={month}/train-*.parquet"

    print(f"Streaming {DATASET} at {args.revision}; target: {args.max_games:,} evaluated games.",
          flush=True)
    games = load_dataset(DATASET, split="train", streaming=True,
                         revision=args.revision, **options)
    scanned, saved = save_evaluated_games(games, args.output, args.max_games)
    print(f"Saved {saved:,} evaluated games to {args.output} (scanned {scanned:,}).")
    if saved < args.max_games:
        print("Selected source exhausted before reaching --max-games.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Complete rows already written are preserved.", file=sys.stderr)
        sys.exit(130)
