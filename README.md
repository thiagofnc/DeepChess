# DeepChess
This is going to be awesome

## Stockfish-annotated games

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
```

Streams [Lichess games](https://huggingface.co/datasets/Lichess/standard-chess-games)
and saves 500 games containing Stockfish `[%eval ...]` annotations by default.
Output: `data/baseline_stockfish_games.jsonl` (ignored by Git), one game per line.
Pawn and mate scores qualify; not every move in a matching game has a score.
Unannotated games are scanned over the network but are not saved.

Optional: `--max-games 500`, `--year 2024 --month 1`, `--output data/sample.jsonl`, or
`--revision <commit-hash>`. Existing output files are never overwritten.
