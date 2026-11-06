# DeepChess

For this project we're training a CNN to guess how good a chess position is. We use Lichess games that already have Stockfish evaluations in them.

The network takes a board position and outputs the estimated evaluation in pawns, from white's perspective (positive is good for White). Positions labeled as forced mates are excluded from training, so the network never sees mate scores.

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

## Using `infer_and_train.py` for other code

Importing the file is clean. Other scripts can use it without trouble. The command line arguments are only used when the file is ran as a script.

## Results so far

The CNN is very accurate and currently gets about 0.6 MSE on the evaluation of a given position. This is very good and indicates it has learned the rules to chess very well.

### Evaluate saved weights

Reports MAE and RMSE (in pawns) for the network and for a constant-value baseline on a test set stored as an `.npz` file with arrays `x` (board tensors) and `y` (evaluations).

```bash
# Same as running with no subcommand
python infer_and_train.py evaluate

# Evaluate a newly trained model
python infer_and_train.py evaluate --weights_path local_data/new_weights.pt
```

Running the script with no arguments, or with only evaluate flags, behaves like `evaluate`.

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--data_path` | `data/test_data_basic.npz` | Test data file |
| `--weights_path` | `data/model_weights.pt` | Weights to load |
| `--batch_size` | `4096` | Batch size for inference |
| `--baseline_value` | `0.0` | Constant prediction used for the baseline comparison |

### Train

Trains the network on a games file, validating after every epoch. Whenever validation RMSE improves, the weights are saved, so the checkpoint on disk is always the best epoch. At the end the best weights are also loaded back into the model in memory.

```bash
# Quick smoke test on a few games
python infer_and_train.py train --max_games 50 --epochs 1

# Full run
python infer_and_train.py train --epochs 20

# Continue from existing weights
python infer_and_train.py train --init_weights data/model_weights.pt --lr 0.002
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--games_path` | `DEFAULT_INPUT` from `chess_data.py` | JSONL games file. The default is `local_data/regressor_games.jsonl` in the parent of the folder containing `chess_data.py`, so pass this explicitly if your data lives elsewhere. |
| `--max_games` | all | Limit the number of games loaded (useful for debugging) |
| `--checkpoint_path` | `local_data/new_weights.pt` | Where the best weights are saved. The folder is created if needed. |
| `--init_weights` | none (random init) | Weights file to start from |
| `--epochs` | `10` | Number of passes over the training set |
| `--batch_size` | `256` | Training batch size |
| `--lr` | `0.01` | Learning rate |
| `--weight_decay` | `1e-4` | L2 regularization strength |
| `--momentum` | `0.9` | SGD momentum |
| `--seed` | `0` | Seed for weight initialization and batch order |

Training details:

- Optimizer is AdamW, and the loss is MSE.
- Gradients are clipped to a norm of 5.0, and training stops with an error if the loss or validation error becomes non-finite.
- Games are split 80/10/10 into train/validation/test **before** positions are extracted, so positions from one game never appear in more than one split. The split uses a fixed seed (42) from `chess_data.py`.
- The test portion of the split is not used or saved during training.
- The baseline reported during training is the mean evaluation of the training set.
- Each epoch prints one JSON line with the epoch number, train MSE, and validation metrics.
- Checkpoints are raw PyTorch state dicts, the same format as `data/model_weights.pt`, so any checkpoint can be loaded by `load_model` or passed to `evaluate --weights_path`.

### Network input

The network takes a float tensor of shape `(batch, 19, 8, 8)` produced by `board_to_array`:

| Planes | Content |
| ------ | ------- |
| 0-5 | White pawn, knight, bishop, rook, queen, king |
| 6-11 | Black pawn, knight, bishop, rook, queen, king |
| 12-15 | Castling rights (White kingside, White queenside, Black kingside, Black queenside) |
| 16 | En passant target square |
| 17 | Side to move (all ones if White) |
| 18 | All ones |

The board is always oriented from White's side and is not flipped for the side to move.

## Directory Structure

`scripts/`: Folder for any python scripts.

`data/`: Folder for any data that should be included in the repo.

`local_data/`: Folder for any data that should not be included in the repo. Does not exist in repo. Training checkpoints are saved here by default.

`tests/`: Folder for testing scripts.

## TODO

- Get new test data from split instead of reusing file. Games may overlap with training games currently.
- Consider different optimizers.
- Can set up next move prediction by checking all possible positions to move to scores and picking whats best (or another strategy)
- A TUI chess board to play would be neat
