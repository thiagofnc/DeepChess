# DeepChess

For this project we're training a CNN to guess how good a chess position is. We use Lichess games that already have Stockfish evaluations in them.

## How to run it

Install the requirements with `pip install -r requirements.txt`. Then run Chess_Board_Maker.ipynb

## Results so far

The CNN is very accurate and currently gets about 0.6 MSE on the evaluation of a given position. This is very good and indicates it has learned the rules to chess very well.
