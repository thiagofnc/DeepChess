# DeepChess

A CNN that learns to evaluate chess positions from Stockfish-annotated Lichess games.

## Setup and data

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py --max-games 500 --year 2013 --month 1 --output data/regressor_games.jsonl
```

Downloads 500 annotated games from [Lichess on Hugging Face](https://huggingface.co/datasets/Lichess/standard-chess-games).
Data is ignored by Git; existing files are never overwritten. Training uses numeric
scores in pawns from White's perspective, skipping mate scores and unannotated positions.

## Training

Run the constant-average baseline with `python scripts/train_regressor.py`.
For the CNN, open `scripts/Chess_Board_Maker.ipynb` from the repository root or
`scripts/` directory and run all cells. Training settings are in the notebook.

Both models use the same 80/10/10 game split (seed 42). The CNN trains for 20 epochs,
selects its best checkpoint using validation RMSE, then evaluates on held-out test games.

## Results

500 games; 50 test games containing 3,108 labeled positions. Epoch 19 was selected.
Errors are in pawns; lower is better.

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Constant average | 4.211 | 10.847 |
| CNN | 4.149 | 10.713 |
| Simple material count | **2.820** | **8.903** |

The CNN slightly beats the constant baseline, but counting material still performs better.

## Tests

```powershell
python -m unittest discover -s tests -v
```
