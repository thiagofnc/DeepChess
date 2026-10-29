"""Shared game splits, PGN labels, and evaluation metrics for both models."""

import io
import json
import math
from pathlib import Path
import random

import chess.pgn


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data/regressor_games.jsonl"


def load_games(path=DEFAULT_INPUT, max_games=None):
    """Read unique games; an optional limit is only for debugging."""
    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be positive or None")
    games, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("movetext"), str):
                raise ValueError(f"Line {line_number}: missing movetext")
            key = record.get("Site") or record["movetext"]
            if key in seen:
                continue
            seen.add(key)
            games.append(record)
            if max_games is not None and len(games) >= max_games:
                break
    return games


def split_games(games, seed=42):
    """Reproducible 80/10/10 split before extracting any positions."""
    if len(games) < 3:
        raise ValueError("At least three unique games are needed for train/validation/test")
    shuffled = list(games)
    random.Random(seed).shuffle(shuffled)
    held_out = max(1, len(shuffled) // 10)
    return {
        "train": shuffled[: -2 * held_out],
        "validation": shuffled[-2 * held_out : -held_out],
        "test": shuffled[-held_out:],
    }


def evaluated_positions(record):
    """Yield resulting boards and numeric pawn scores from White's perspective.

    Every mainline move is played, including moves without annotations. Mate
    distances are not pawn values and are excluded from both models' targets.
    """
    game = chess.pgn.read_game(io.StringIO(record["movetext"]))
    if game is None or game.errors:
        raise ValueError(f"Invalid PGN in {record.get('Site', 'game')}")
    board = game.board()
    for node in game.mainline():
        board.push(node.move)
        evaluation = node.eval()
        if evaluation is None or evaluation.is_mate():
            continue
        score = evaluation.white().score()
        if score is not None:
            yield board.copy(stack=False), score / 100.0


def numeric_scores(games):
    return [score for game in games for _, score in evaluated_positions(game)]


def regression_metrics(actual, predicted):
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Metrics require nonempty, equally sized labels and predictions")
    errors = [p - y for y, p in zip(actual, predicted)]
    return {
        "mae_pawns": sum(abs(e) for e in errors) / len(errors),
        "rmse_pawns": math.sqrt(sum(e * e for e in errors) / len(errors)),
    }
