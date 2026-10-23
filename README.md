# DeepChess
This is going to be awesome

## Stockfish-annotated games

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py
```

Streams [Lichess games](https://huggingface.co/datasets/Lichess/standard-chess-games)
and saves all games containing Stockfish `[%eval ...]` annotations, with no limit.
Output: `data/stockfish_games.jsonl` (ignored by Git), one complete game per line.
Pawn and mate scores qualify; not every move in a matching game has a score.
Unannotated games are scanned over the network but are not saved.

Optional: `--max-games 10000`, `--year 2024 --month 1`, `--output data/sample.jsonl`, or
`--revision <commit-hash>`. Existing output files are never overwritten.
