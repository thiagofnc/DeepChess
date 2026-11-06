"""Stream Lichess games and save only games containing Stockfish evaluations."""

import argparse
from datetime import date, datetime, time
import json
from pathlib import Path
import re
import sys


DATASET = "Lichess/standard-chess-games"
DEFAULT_MAX_GAMES = 500
# Accept pawn scores and mate scores, including optional analysis depth.
EVAL_PATTERN = re.compile(
    r"\[%eval\s+(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)|#[+-]?\d+)"
    r"(?:\s*,\s*\d+)?\s*\]"
)
SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*$", re.IGNORECASE)
SIZE_UNITS = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def size_bytes(value):
    """Parse sizes like '500MB', '2G', '1.5GB', '250000' (binary units: 1 MB = 1024**2 bytes)."""
    match = SIZE_PATTERN.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}; use e.g. 500MB, 2GB, or a byte count"
        )
    number = int(float(match.group(1)) * SIZE_UNITS[match.group(2).lower()])
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1024
    return f"{size:,.2f} TB"


def json_default(value):
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_evaluated_games(games, output, max_games=None, max_bytes=None,
                         progress_every=10_000):
    """Save complete matching rows as JSONL; never overwrite an existing file.

    Stops when max_games rows are saved or when writing the next row would push
    the file past max_bytes (whichever limit is set). Rows are never truncated.

    Returns (scanned, saved, bytes_written, limit_reached).
    """
    scanned = saved = written = 0
    limit_reached = False
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the byte count accurate on Windows.
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for game in games:
            scanned += 1
            if "movetext" not in game:
                raise ValueError("Dataset row has no 'movetext' field; check its schema.")
            movetext = game["movetext"]
            if isinstance(movetext, str) and EVAL_PATTERN.search(movetext):
                line = json.dumps(game, default=json_default, ensure_ascii=False) + "\n"
                line_size = len(line.encode("utf-8"))
                if max_bytes is not None and written + line_size > max_bytes:
                    limit_reached = True
                    break
                handle.write(line)
                written += line_size
                saved += 1
                if max_games is not None and saved >= max_games:
                    limit_reached = True
                    break
            if scanned % progress_every == 0:
                print(f"Scanned {scanned:,} games; saved {saved:,} "
                      f"({format_size(written)}).", flush=True)
    return scanned, saved, written, limit_reached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument("--max-games", type=positive_int,
                       help=f"Number of matching games to save "
                            f"(default: {DEFAULT_MAX_GAMES} if --max-size is not set).")
    limit.add_argument("--max-size", type=size_bytes,
                       help="Maximum output file size, e.g. 500MB or 2GB "
                            "(1 MB = 1024**2 bytes). Alternative to --max-games.")
    parser.add_argument("--output", type=Path,
                        default=Path("local_data/regressor_games.jsonl"))
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
    if args.max_games is None and args.max_size is None:
        args.max_games = DEFAULT_MAX_GAMES

    try:
        from datasets import load_dataset
    except ImportError:
        parser.exit(1, "Install dependencies first: python -m pip install -r requirements.txt\n")

    options = {}
    if args.year is not None:
        month = f"{args.month:02d}" if args.month is not None else "*"
        options["data_files"] = f"local_data/year={args.year}/month={month}/train-*.parquet"

    if args.max_size is not None:
        target = f"up to {format_size(args.max_size)} of evaluated games"
    else:
        target = f"{args.max_games:,} evaluated games"
    print(f"Streaming {DATASET} at {args.revision}; target: {target}.", flush=True)
    games = load_dataset(DATASET, split="train", streaming=True,
                         revision=args.revision, **options)
    scanned, saved, written, limit_reached = save_evaluated_games(
        games, args.output, args.max_games, args.max_size
    )
    print(f"Saved {saved:,} evaluated games ({format_size(written)}) to {args.output} "
          f"(scanned {scanned:,}).")
    if not limit_reached:
        limit_name = "--max-size" if args.max_size is not None else "--max-games"
        print(f"Selected source exhausted before reaching {limit_name}.", file=sys.stderr)
    elif saved == 0:
        print("Warning: the first matching game is larger than --max-size; nothing saved.",
              file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Complete rows already written are preserved.", file=sys.stderr)
        sys.exit(130)