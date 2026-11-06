import json
from copy import deepcopy
from pathlib import Path
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys

from chess_data import (DEFAULT_INPUT, evaluated_positions, load_games, regression_metrics, split_games)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"


def board_to_array(board):
    """Convert a chess.Board to a 19 x 8 x 8 array for the network."""
    # Create the 19 x 8 x 8 array
    x = np.zeros((19, 8, 8), dtype=np.uint8)
    
    # Layers 0-11: Piece positions
    # White: pawn, knight, bishop, rook, queen, king
    # Black: pawn, knight, bishop, rook, queen, king
    piece_layer = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,
        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        layer = piece_layer[(piece.color, piece.piece_type)]
        x[layer, row, col] = 1

    # Layers 12-15: Castling rights
    x[12, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
    x[13, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
    x[14, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
    x[15, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))

    # Layer 16: Location of a pawn that can legally be
    # en passant
    if board.ep_square is not None:
        ep_square = board.ep_square
        row = 7 - chess.square_rank(ep_square)
        col = chess.square_file(ep_square)
        x[16, row, col] = 1

    # Layer 17: Side to move
    x[17, :, :] = int(board.turn == chess.WHITE)

    # Board shape
    x[18, :, :] = 1

    return x


#residual block for a single residual layer of the netwrok
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x):
        #here we just just use the basic convoluition then batch normalize then relu, and do that twice and add it since its a residual network
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


#full model with each residual block
class deepChess(nn.Module):
    def __init__(self):
        super(deepChess, self).__init__()

        #input convolution layer
        self.conv_in = nn.Conv2d(19, 128, kernel_size=3, padding='same', bias=False)
        self.bn_in = nn.BatchNorm2d(128)

        #here we do 7 resblocks from earlier in a row (14 total convolution layers)
        self.blocks = nn.Sequential(*[ResBlock() for _ in range(7)])
        self.conv_head = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm2d(32)

        #our two mlp layers at the end
        self.fc1 = nn.Linear(32*8*8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, a0):
        #first input conv
        a = F.relu(self.bn_in(self.conv_in(a0)))
        #run through each residual block
        a = self.blocks(a)
        a = F.relu(self.bn_head(self.conv_head(a)))

        #flattened output so we can run the mlp on them
        flat = a.view(a.shape[0], 32*8*8)
        return self.fc2(F.gelu(self.fc1(flat))) * 10


def load_model(weights_path):
    model = deepChess().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def evaluate(model, loader, baseline_value):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for features, targets in loader:
            actual.extend(targets.flatten().tolist())
            predicted.extend(model(features.to(device)).flatten().tolist())
    return {
        "cnn": regression_metrics(actual, predicted),
        "baseline": regression_metrics(actual, [baseline_value] * len(actual)),
    }


def train_model(model, optimizer, criterion, train_loader, validation_loader, epochs, baseline_value, checkpoint_path=None):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    best_rmse = float("inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for features, targets in train_loader:
            # Move features and targets to the appropriate device (CPU or GPU)
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(targets)
        validation = evaluate(model, validation_loader, baseline_value)
        rmse = validation["cnn"]["rmse_pawns"]
        if not np.isfinite(rmse):
            raise RuntimeError("Non-finite validation error; check labels and learning rate")
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = deepcopy(model.state_dict())
            if checkpoint_path is not None:
                # make the checkpoint folder if it doesnt exist
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                # Previously used checkpoint format
                # torch.save({"model_state_dict": best_state, "epoch": epoch + 1,
                #             "validation": validation}, checkpoint_path)
                # New checkpoint format to match the inferance loader
                torch.save(best_state, checkpoint_path)
        history.append({"epoch": epoch + 1,
                        "train_mse": total_loss / len(train_loader.dataset),
                        "validation": validation})
        print(json.dumps(history[-1]))
    # Select the checkpoint using validation only; test games are not used here.
    model.load_state_dict(best_state)
    return history


def make_dataset(games):
    arrays, labels = [], []
    for record in games:
        for board, score in evaluated_positions(record):
            arrays.append(board_to_array(board))
            labels.append(score)
    if not labels:
        raise ValueError("A split has no numeric evaluation labels")
    features = torch.from_numpy(np.stack(arrays)).float()
    targets = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    return TensorDataset(features, targets), labels


def run_inference(args):
    """Runs inference based on provided arguments."""
    # Inference data: x is board position, y is eval
    data = np.load(args.data_path)
    test_set = TensorDataset(torch.from_numpy(data["x"]).float(),
                            torch.from_numpy(data["y"]).float())
    
    loader = DataLoader(test_set, batch_size=args.batch_size)
    model = load_model(args.weights_path)
        
    # Run once after selecting the model; do not tune hyperparameters on test error.
    test_metrics = evaluate(model, loader, args.baseline_value)
    print("Held-out test metrics (pawns):")
    print(json.dumps(test_metrics, indent=2))

    print("cost (MSE):", test_metrics["cnn"]["rmse_pawns"] ** 2)
    print("average eval error (pawns):", test_metrics["cnn"]["mae_pawns"])


def run_training(args):
    torch.manual_seed(args.seed)

    # Load some number of games
    games = load_games(args.games_path, max_games=args.max_games)
    # Split the games into train, validation, and test, 
    # nothing is done with the test portion in training
    # TODO: Whoever wrote chess_data.py should figure out
    # if keeping a portion of the data for test only is actually what we want
    # atleast the test portion should be saved for a file so we use it during inference or something.
    splits = split_games(games)

    # Build tensors for train and validation
    # the "labels" are kinda reduntant since the scorings are already in the tensor,
    # but they make it easy to get the baseline
    train_set, train_labels = make_dataset(splits["train"])
    val_set, _ = make_dataset(splits["validation"])

    # Get the baseline value to compare to.
    baseline = sum(train_labels) / len(train_labels)
    # Shows how many entries are being trained on and validated with and what the baseline is
    print(f"train positions: {len(train_set)}, validation positions: {len(val_set)}, "
          f"baseline (train mean): {baseline:.4f}")

    # Build the torch data loaders.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # If there are already weights given, start there, otherwise init a fresh model. 
    # Makes sure to move everything to the correct device
    model = load_model(args.init_weights) if args.init_weights else deepChess().to(device)

    # Using AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Use MSE loss.
    criterion = nn.MSELoss()

    # Train
    train_model(model, optimizer, criterion, train_loader, val_loader,
                args.epochs, baseline, checkpoint_path=args.checkpoint_path)
    print("Done.")


def build_parser():
    parser = argparse.ArgumentParser(description="DeepChess evaluation network")
    sub = parser.add_subparsers(dest="command")

    ev = sub.add_parser("evaluate", help="Report test-set metrics for saved weights")
    ev.add_argument("--data_path", default="data/test_data_basic.npz")
    ev.add_argument("--weights_path", default="data/model_weights.pt")
    ev.add_argument("--batch_size", type=int, default=4096)
    ev.add_argument("--baseline_value", type=float, default=0.0)

    tr = sub.add_parser("train", help="Train the network and save the best checkpoint")
    tr.add_argument("--games_path", default=str(DEFAULT_INPUT))
    tr.add_argument("--max_games", type=int, default=None)
    tr.add_argument("--checkpoint_path", default="local_data/new_weights.pt")
    tr.add_argument("--init_weights", default=None, help="Optional weights file to start from instead of random init")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch_size", type=int, default=1024)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    # tr.add_argument("--momentum", type=float, default=0.9)
    tr.add_argument("--seed", type=int, default=0)
    

    return parser


COMMANDS = {"evaluate": run_inference, "train": run_training}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # If a subcommand is not given just default to acting like before
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["evaluate", *argv]
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)

if __name__ == "__main__":
    main()