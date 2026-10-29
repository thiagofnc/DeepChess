import json
from pathlib import Path
import sys
import tempfile
import unittest

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chess_data import evaluated_positions, load_games, regression_metrics, split_games
from train_regressor import fit_and_evaluate


class ChessDataTests(unittest.TestCase):
    def test_missing_annotations_do_not_shift_boards_or_labels(self):
        samples = list(evaluated_positions({"movetext":
            "1. e4 { [%eval 0.2] [%clk 0:05:00] } e5 "
            "2. Nf3 { [%eval 0.3,18] } *"}))
        self.assertEqual([score for _, score in samples], [0.2, 0.3])
        board = chess.Board()
        board.push_san("e4")
        self.assertEqual(samples[0][0].fen(), board.fen())
        board.push_san("e5")
        board.push_san("Nf3")
        self.assertEqual(samples[1][0].fen(), board.fen())

    def test_mate_and_variations_are_not_numeric_mainline_labels(self):
        samples = list(evaluated_positions({"movetext":
            "1. e4 { [%eval #-1] } (1. d4 { [%eval 9.0] }) "
            "e5 { [%eval -0.25] } *"}))
        self.assertEqual([score for _, score in samples], [-0.25])
        self.assertTrue(samples[0][0].turn == chess.WHITE)

    def test_split_is_reproducible_disjoint_and_does_not_mutate_input(self):
        games = [{"Site": str(i)} for i in range(20)]
        original = games.copy()
        splits = split_games(games)
        self.assertEqual(splits, split_games(games))
        self.assertEqual(games, original)
        self.assertEqual([len(g) for g in splits.values()], [16, 2, 2])
        all_ids = [g["Site"] for group in splits.values() for g in group]
        self.assertEqual(len(set(all_ids)), len(games))

    def test_loads_all_games_by_default_and_deduplicates(self):
        records = [{"Site": str(i), "movetext": "1. e4 *"} for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records + records), encoding="utf-8")
            self.assertEqual(len(load_games(path)), 5)
            self.assertEqual(len(load_games(path, max_games=3)), 3)
            with self.assertRaises(ValueError):
                load_games(path, max_games=0)

    def test_baseline_does_not_fit_held_out_labels(self):
        records = [{"Site": str(i), "movetext": f"1. e4 {{ [%eval {i}] }} *"}
                   for i in range(10)]
        splits = split_games(records)
        expected_mean = sum(int(r["Site"]) for r in splits["train"]) / len(splits["train"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            def write():
                path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            write()
            before = fit_and_evaluate(path)
            self.assertEqual(before["mean_eval_pawns"], expected_mean)
            for record in splits["test"] + splits["validation"]:
                record["movetext"] = "1. e4 { [%eval 100] } *"
            write()
            after = fit_and_evaluate(path)
            self.assertEqual(before["mean_eval_pawns"], after["mean_eval_pawns"])
            self.assertNotEqual(before["metrics"], after["metrics"])

    def test_metrics_and_small_dataset_errors(self):
        metrics = regression_metrics([1, 3], [0, 1])
        self.assertEqual(metrics["mae_pawns"], 1.5)
        self.assertAlmostEqual(metrics["rmse_pawns"], 2.5 ** 0.5)
        with self.assertRaises(ValueError):
            regression_metrics([], [])
        with self.assertRaises(ValueError):
            split_games([{}, {}])


if __name__ == "__main__":
    unittest.main()
