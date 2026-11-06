// Copies the big runtime files (ONNX Runtime wasm, Stockfish) from node_modules into public/vendor,
// so they are served as plain static files next to the site instead of being bundled.
import { cpSync, mkdirSync, existsSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const copy = (from, to, names) => {
  mkdirSync(join(root, to), { recursive: true });
  for (const n of names) cpSync(join(root, from, n), join(root, to, n));
};

rmSync(join(root, 'public/vendor'), { recursive: true, force: true });
copy('node_modules/onnxruntime-web/dist', 'public/vendor/ort', [
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
]);
// Stockfish is GPL-3.0: its license ships next to it.
copy('node_modules/stockfish/bin', 'public/vendor/stockfish', [
  'stockfish-19-lite-single.js',
  'stockfish-19-lite-single.wasm',
]);
copy('node_modules/stockfish', 'public/vendor/stockfish', ['Copying.txt']);
if (!existsSync(join(root, 'public/model/deepchess.onnx'))) {
  console.warn('public/model/deepchess.onnx is missing: run `python web/export/export_onnx.py` from the repo root.');
}
