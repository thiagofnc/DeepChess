# DeepChess web

A browser front end for the DeepChess network: play against it, watch it play itself or Stockfish, set up any position, and look inside the model. Everything runs on the client; there is no backend.

- **Network inference:** [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) with the WebGPU backend (falls back to WebAssembly when WebGPU is unavailable), in a Web Worker together with the move search, so the page never blocks on it.
- **Rules:** [chess.js](https://github.com/jhlywa/chess.js).
- **Opponent engine:** Stockfish 19 (lite, single-threaded WASM) in a Web Worker, loaded the first time you pick it.
- **UI:** vanilla TypeScript + Vite, no framework.

## Features

| Tab | What you get |
| --- | --- |
| Play | Each colour can be you, DeepChess (looking 1 or 2 moves ahead) or Stockfish at ~1320–3190 Elo; pick a bot for both to watch them play. Optional evaluation bar, eval history graph, and the full list of the bot's moves ranked by its own evaluation, with arrows on the board. A position editor builds any legal board state, with FEN import/export. |
| Explore the network | Step through the game and see the input planes, every layer's feature maps (128 channels per residual block, with mean/max/per-channel overlays on the board), the 256 hidden units and how much each pushes the evaluation, how much each block changes the representation, and a piece-impact (occlusion) map. |

### How the bot picks a move

The network scores a *position* (pawns, from White's point of view). To move, DeepChess plays each legal move, scores every resulting position in one GPU batch, and picks the best for the side to move (the highest for White, the lowest for Black). The 2-ply mode also considers every reply (full-width minimax). Checkmate and draws are handled by the rules engine, because the network never saw mate scores in training. It always plays its top-ranked move, so a game between two identical DeepChess bots plays out the same way every time.

## Running it

```bash
# 1. (only if the weights changed) export the model for the browser; run from the repo root
pip install onnx onnxruntime onnxscript   # on top of requirements.txt
python web/export/export_onnx.py          # writes web/public/model/{deepchess.onnx,golden.json,fc2.json}

# 2. the site
cd web
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in web/dist, works from any sub-path
npm test           # parity tests, see below
```

`web/public/model/deepchess.onnx` is committed so the site works without Python. `npm run dev/build` copies the ONNX Runtime wasm and Stockfish from `node_modules` into `public/vendor/` (git-ignored).

`export_onnx.py` imports the model and `board_to_array` straight from `scripts/infer_and_train.py`, so the browser model cannot drift from the training code. The exported graph exposes every intermediate activation as an output; during play only `eval` is requested.

## Tests

`npm test` loads `public/model/golden.json` (written by the export script from the PyTorch model) and checks that

1. the TypeScript board encoder produces **exactly** the same 19×8×8 planes as `board_to_array`, including the python-chess convention of setting the en-passant square after every double pawn push, and
2. the ONNX model run through onnxruntime-web reproduces the PyTorch evaluations (to ~1e-3).

## Hosting

The build is plain static files with relative URLs, so it can be served from anywhere (GitHub Pages, any static host, or a sub-path of another site). The `.wasm` files must be served as `application/wasm`. Cross-origin isolation is not required.

Stockfish is GPL-3.0; its license is copied next to it in `vendor/stockfish/Copying.txt`.
