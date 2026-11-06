"""Export data/model_weights.pt to ONNX for the browser, plus a golden file for parity tests.

Run from the repo root:  python web/export/export_onnx.py
Writes web/public/model/deepchess.onnx and web/public/model/golden.json
"""

import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import infer_and_train as it  # noqa: E402  (reuses the real model + encoder, so they can't drift)

OUT = ROOT / "web" / "public" / "model"
WEIGHTS = ROOT / "data" / "model_weights.pt"

N_BLOCKS = len(it.deepChess().blocks)
OUTPUT_NAMES = ["stem"] + [f"block{i}" for i in range(N_BLOCKS)] + ["head", "fc1", "eval"]


class Traced(torch.nn.Module):
    """Same forward as deepChess, but returns every stage so the browser can show them."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        m = self.m
        a = F.relu(m.bn_in(m.conv_in(x)))
        outs = [a]
        for b in m.blocks:
            a = b(a)
            outs.append(a)
        head = F.relu(m.bn_head(m.conv_head(a)))
        fc1 = F.gelu(m.fc1(head.flatten(1)))
        outs += [head, fc1, m.fc2(fc1) * 10]
        return tuple(outs)


FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # ep square set, no legal capture
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # legal ep capture
    "r3k2r/pppq1ppp/2n1bn2/3p4/3P4/2N1BN2/PPPQ1PPP/R3K2R w Kq - 4 9",  # partial castling
    "8/5pk1/6p1/8/3B4/6P1/5PK1/8 b - - 0 40",  # endgame, black to move
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # mate threat
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = it.load_model(WEIGHTS).cpu().eval()
    traced = Traced(model).eval()
    dummy = torch.zeros(1, 19, 8, 8)
    torch.onnx.export(
        traced, (dummy,), str(OUT / "deepchess.onnx"),
        input_names=["planes"], output_names=OUTPUT_NAMES,
        dynamic_axes={n: {0: "batch"} for n in ["planes"] + OUTPUT_NAMES},
        opset_version=20, dynamo=False,
    )

    golden = []
    with torch.no_grad():
        for fen in FENS:
            x = it.board_to_array(chess.Board(fen))
            outs = traced(torch.from_numpy(x).float()[None])
            ref = model(torch.from_numpy(x).float()[None]).item()
            assert abs(ref - outs[-1].item()) < 1e-4
            golden.append({
                "fen": fen,
                "planes": x.flatten().tolist(),
                "eval": outs[-1].item(),
                "sums": {n: float(o.sum()) for n, o in zip(OUTPUT_NAMES, outs)},
            })
    (OUT / "golden.json").write_text(json.dumps({"outputs": OUTPUT_NAMES, "cases": golden}))
    # Output layer, so the browser can show how much each of the 256 hidden units pushes the evaluation.
    fc2 = {"weights": (model.fc2.weight[0] * 10).tolist(), "bias": float(model.fc2.bias[0]) * 10}
    (OUT / "fc2.json").write_text(json.dumps(fc2))
    print("exported", OUT / "deepchess.onnx", (OUT / "deepchess.onnx").stat().st_size // 1024, "KB")

    # Cross-check with onnxruntime on CPU.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT / "deepchess.onnx"))
    for g in golden:
        got = sess.run(["eval"], {"planes": np.array(g["planes"], np.float32).reshape(1, 19, 8, 8)})[0].item()
        assert abs(got - g["eval"]) < 1e-3, (g["fen"], got, g["eval"])
    print("onnxruntime parity ok;", [round(g["eval"], 2) for g in golden])


if __name__ == "__main__":
    main()
