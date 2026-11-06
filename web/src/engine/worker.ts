// Runs the network and the move search off the main thread, so the page stays smooth however long they take.
import * as ort from 'onnxruntime-web';
import { Chess } from 'chess.js';
import { PLANE_SIZE, encode } from './encode';
import { analyze } from './search';
import { STAGES, type Backend, type Request, type Response } from './protocol';

/** PyTorch's evaluation of the start position: a backend that disagrees is broken and not used. */
const START_EVAL = 0.22404715418815613;
/**
 * WebGPU compiles shaders per tensor shape, so on the GPU every batch has one of two sizes (1 or this, padded).
 * Otherwise each new number of legal moves would stall on a recompile.
 */
const GPU_BATCH = 64;
const WASM_BATCH = 16;

const post = (m: Response, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

class Aborted extends Error {}

let session: ort.InferenceSession;
let backend: Backend;
let latestAnalyze = 0;

// Only one run at a time may use the session.
let lock: Promise<unknown> = Promise.resolve();
function run(feeds: ort.InferenceSession.FeedsType, fetches: string[]): Promise<ort.InferenceSession.ReturnType> {
  const r = lock.then(() => session.run(feeds, fetches));
  lock = r.catch(() => {});
  return r;
}

async function evaluate(planes: Float32Array, n: number, stillWanted: () => boolean = () => true): Promise<Float32Array> {
  const out = new Float32Array(n);
  if (n === 1) {
    const res = await run({ planes: new ort.Tensor('float32', planes.slice(0, PLANE_SIZE), [1, 19, 8, 8]) }, ['eval']);
    out[0] = (res.eval.data as Float32Array)[0];
    return out;
  }
  const size = backend === 'webgpu' ? GPU_BATCH : WASM_BATCH;
  for (let i = 0; i < n; i += size) {
    if (!stillWanted()) throw new Aborted();
    const m = Math.min(size, n - i);
    const count = backend === 'webgpu' ? size : m;
    const batch = new Float32Array(count * PLANE_SIZE);
    batch.set(planes.subarray(i * PLANE_SIZE, (i + m) * PLANE_SIZE));
    const res = await run({ planes: new ort.Tensor('float32', batch, [count, 19, 8, 8]) }, ['eval']);
    out.set((res.eval.data as Float32Array).subarray(0, m), i);
  }
  return out;
}

async function init(base: string, backends: Backend[]): Promise<Backend> {
  const here = (rel: string) => new URL(rel, base).href;
  ort.env.wasm.wasmPaths = { mjs: here('vendor/ort/ort-wasm-simd-threaded.jsep.mjs'), wasm: here('vendor/ort/ort-wasm-simd-threaded.jsep.wasm') };
  ort.env.wasm.numThreads = 1; // no cross-origin isolation => no SharedArrayBuffer
  post({ id: -1, status: 'Downloading weights…' });
  const model = new Uint8Array(await (await fetch(here('model/deepchess.onnx'))).arrayBuffer());
  let lastErr: unknown = new Error('No backend available');
  for (const p of backends) {
    try {
      post({ id: -1, status: p === 'webgpu' ? 'Starting WebGPU…' : 'Starting WebAssembly…' });
      session = await ort.InferenceSession.create(model, { executionProviders: [p], graphOptimizationLevel: 'all' });
      backend = p;
      const got = (await evaluate(encode(new Chess()), 1))[0];
      if (!(Math.abs(got - START_EVAL) < 0.05)) throw new Error(`${p} gave ${got} for the start position, expected ${START_EVAL}`);
      if (p === 'webgpu') await evaluate(new Float32Array(2 * PLANE_SIZE), 2); // compile the batch shaders before the first move
      return p;
    } catch (e) {
      lastErr = e;
      console.warn(`${p} backend failed`, e);
    }
  }
  throw lastErr;
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const req = e.data;
  const { id } = req;
  try {
    if (req.type === 'init') {
      post({ id, ok: true, result: await init(req.base, req.backends) });
    } else if (req.type === 'analyze') {
      latestAnalyze = id; // a newer position makes any search still running pointless
      const chess = new Chess(req.startFen);
      for (const san of req.sans) chess.move(san);
      const result = await analyze((p, n) => evaluate(p, n, () => latestAnalyze === id), chess, req.depth, req.wantMoves);
      post({ id, ok: true, result });
    } else if (req.type === 'evaluate') {
      post({ id, ok: true, result: await evaluate(req.planes, req.n) });
    } else {
      const res = await run({ planes: new ort.Tensor('float32', req.planes, [1, 19, 8, 8]) }, [...STAGES]);
      const result: Record<string, Float32Array> = { planes: req.planes };
      for (const s of STAGES) result[s] = new Float32Array(res[s].data as Float32Array);
      post({ id, ok: true, result }, STAGES.map((s) => result[s].buffer as ArrayBuffer));
    }
  } catch (err) {
    if (err instanceof Aborted) post({ id, ok: true, result: null });
    else post({ id, ok: false, error: String((err as Error).message ?? err) });
  }
};
