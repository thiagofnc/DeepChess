import json
from copy import deepcopy
from pathlib import Path
 
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
 
from chess_data import evaluated_positions, regression_metrics


# Define paths and constants
DATA_PATH = "data/test_data_basic.npz"
WEIGHTS_PATH = "data/model_weights.pt"
TRAINING_WEIGHTS_PATH = "data/new_weights.pt"
BATCH_SIZE = 4096
# TODO: Replace actual mean with the mean of the training set evaluations
BASELINE_VALUE = 0.0

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


def evaluate(model, loader):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for features, targets in loader:
            actual.extend(targets.flatten().tolist())
            predicted.extend(model(features.to(device)).flatten().tolist())
    return {
        "cnn": regression_metrics(actual, predicted),
        "baseline": regression_metrics(actual, [BASELINE_VALUE] * len(actual)),
    }


def train_model(model, optimizer, criterion, train_loader, validation_loader, epochs, checkpoint_path=None):
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
        validation = evaluate(model, validation_loader)
        rmse = validation["cnn"]["rmse_pawns"]
        if not np.isfinite(rmse):
            raise RuntimeError("Non-finite validation error; check labels and learning rate")
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = deepcopy(model.state_dict())
            if checkpoint_path is not None:
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



# Inference data: x is board position, y is eval
data = np.load(DATA_PATH)
test_set = TensorDataset(torch.from_numpy(data["x"]).float(),
                         torch.from_numpy(data["y"]).float())
 
loaders = {"test": DataLoader(test_set, batch_size=BATCH_SIZE)}
 
model = load_model(WEIGHTS_PATH)
 
# TODO (training, not fixed yet)
 
 
# Run once after selecting the model; do not tune hyperparameters on test error.
test_metrics = evaluate(model, loaders["test"])
print("Held-out test metrics (pawns):")
print(json.dumps(test_metrics, indent=2))

print("cost (MSE):", test_metrics["cnn"]["rmse_pawns"] ** 2)
print("average eval error (pawns):", test_metrics["cnn"]["mae_pawns"])
