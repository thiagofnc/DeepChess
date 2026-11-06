// Checks the browser pipeline against the python reference (golden.json from export_onnx.py):
//  (a) the JS encoder produces exactly the same planes as scripts/infer_and_train.py board_to_array
//  (b) the ONNX model run through onnxruntime-web (wasm) reproduces the PyTorch evaluations
import { readFileSync } from 'node:fs';
import { Chess } from 'chess.js';
import * as ort from 'onnxruntime-web';
import { PLANE_SIZE, encode, encodeInto, epAfterMove } from '../src/engine/encode.ts';

const golden = JSON.parse(readFileSync(new URL('../public/model/golden.json', import.meta.url), 'utf8'));
const model = readFileSync(new URL('../public/model/deepchess.onnx', import.meta.url));
const dist = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url).href;
ort.env.wasm.wasmPaths = { mjs: dist + 'ort-wasm-simd-threaded.asyncify.mjs', wasm: dist + 'ort-wasm-simd-threaded.asyncify.wasm' };
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(new Uint8Array(model));

let failed = 0;
for (const g of golden.cases) {
  // Replay "start + e4" the way the browser does so the python ep-square convention is exercised.
  const chess = new Chess();
  if (g.fen.startsWith('rnbqkbnr/pppppppp/8/8/4P3')) chess.move('e4');
  else chess.load(g.fen);
  const planes = encode(chess);
  const diff = planes.reduce((n, v, i) => n + (v !== g.planes[i] ? 1 : 0), 0);
  const out = await session.run({ planes: new ort.Tensor('float32', planes, [1, 19, 8, 8]) });
  const err = Math.abs((out.eval.data as Float32Array)[0] - g.eval);
  const ok = diff === 0 && err < 1e-3;
  if (!ok) failed++;
  console.log(ok ? 'ok  ' : 'FAIL', g.fen, `planesDiff=${diff}`, `evalErr=${err.toExponential(2)}`);
}
// The search encodes children with the last move's ep square instead of replaying the game: it must agree.
const game = new Chess();
for (const san of ['e4', 'd5', 'e5', 'f5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'O-O', 'axb5']) game.move(san);
for (const m of game.moves({ verbose: true })) {
  game.move(m);
  const fast = new Float32Array(PLANE_SIZE);
  encodeInto(game, fast, 0, epAfterMove(m));
  const slow = encode(game);
  if (fast.some((v, i) => v !== slow[i])) {
    failed++;
    console.log('FAIL fast encoding differs after', m.san);
  }
  game.undo();
}

if (failed) process.exit(1);
console.log('all parity checks passed');
