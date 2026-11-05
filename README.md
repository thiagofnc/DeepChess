# DeepChess

For this project we're training a CNN to guess how good a chess position is. We use Lichess games that already have Stockfish evaluations in them.

## How to run it

Install the requirements with `pip install -r requirements.txt`. Run all scripts from root.

## Script descriptions and usage

| Script | Description | How to Use |
| ------ | ----------- | ---------- |
| `infer_and_train.py` | Main script for coordinating the inference and training of the network. | Run the script with no arguments. Variables are currently changed directly in the file |
| `download_data.py` | Script for download data to use from hugging face. | Run the script with arguments. Run with "-h" to see aguments. |
| `show_record.py` | Simple script to check one record in raw data. | Modify variable in file to desired record and run with no arguments. |
| `train_sbr.py` | Finds the mean baseline (simple bias regressor output) of a dataset. | Modify variable in file to desired record and run with no arguments. |
| `chess_data.py` | Helper functions for `infer_and_train.py`. | N/A |
| `Chess_Board_Maker.ipynb` | Not currently used. Previous version of `infer_and_train.py` | Notebook |
## Results so far

The CNN is very accurate and currently gets about 0.6 MSE on the evaluation of a given position. This is very good and indicates it has learned the rules to chess very well.


## Directory Structure

`scripts/`: Folder for any python scripts.

`data/`: Folder for any data that should be included in the repo.

`local_data/`: Folder for any data that should not be included in the repo. Does not exist in repo.

`tests/`: Folder for testing scripts.


## TODO

- Fix training. It is partially included in infer_and_train.py but doesn't currently work.
- Add an arg parser to infer_and_train.py which allows selection of training or inference. Or make it integrated with gui controls.
- Fix the baseline in the infer_and_train.py. Theres a variable which defines a baseline as zero at the beggining. Not sure if its needed.
- Remove Chess_Board_Maker.ipynb if no longer needed.