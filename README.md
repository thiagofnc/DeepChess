# DeepChess

For this project we're training a CNN to guess how good a chess position is. We use Lichess games that already have Stockfish evaluations in them.

## How to run it

Install the requirements with `pip install -r requirements.txt`. Then run `scripts/download_data.py` 

To run the simple baseline (average score), use `python scripts/train_regressor.py`. The CNN is in `scripts/Chess_Board_Maker.ipynb`.

## Results so far

The CNN does a little better than guessing the average, but just counting material still beats it
