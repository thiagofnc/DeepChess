import type { Activations, Analysis, Backend, Request, Response } from './protocol';

export { STAGES } from './protocol';
export type { Activations, Analysis, Backend, ScoredMove, Stage } from './protocol';

// Distributive, so each request type keeps its own fields once `id` is removed.
type Payload = Request extends infer R ? (R extends unknown ? Omit<R, 'id'> : never) : never;

/** Page-side handle on the network, which lives in a Web Worker (see worker.ts). */
export class Net {
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  backend: Backend = 'wasm';

  private constructor(private worker: Worker, onStatus: (s: string) => void) {
    worker.onmessage = (e: MessageEvent<Response>) => {
      const m = e.data;
      if ('status' in m) return onStatus(m.status);
      const p = this.pending.get(m.id);
      this.pending.delete(m.id);
      if (m.ok) p?.resolve(m.result);
      else p?.reject(new Error(m.error));
    };
    worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message));
      this.pending.clear();
    };
  }

  private call<T>(req: Payload): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...req, id });
    });
  }

  static async load(onStatus: (s: string) => void = () => {}): Promise<Net> {
    const start = async (backends: Backend[], timeoutMs: number) => {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      const net = new Net(worker, onStatus);
      let timer = 0;
      try {
        net.backend = await Promise.race([
          net.call<Backend>({ type: 'init', base: document.baseURI, backends }),
          new Promise<never>((_, rej) => (timer = window.setTimeout(() => rej(new Error('start-up timed out')), timeoutMs))),
        ]);
        return net;
      } catch (e) {
        worker.terminate(); // a stuck GPU driver can only be escaped by throwing the whole worker away
        throw e;
      } finally {
        clearTimeout(timer);
      }
    };
    if ((navigator as any).gpu) {
      try {
        return await start(['webgpu'], 45000);
      } catch (e) {
        console.warn('WebGPU unavailable, using WebAssembly', e);
      }
    }
    return start(['wasm'], 180000);
  }

  /** Null if a newer analyze() superseded this one. */
  analyze(startFen: string, sans: string[], depth: 1 | 2, wantMoves: boolean): Promise<Analysis | null> {
    return this.call({ type: 'analyze', startFen, sans, depth, wantMoves });
  }

  /** Evaluations in pawns from White's point of view for n positions (planes concatenated). */
  evaluate(planes: Float32Array, n: number): Promise<Float32Array> {
    return this.call({ type: 'evaluate', planes, n });
  }

  /** Every intermediate activation for one position. */
  activations(planes: Float32Array): Promise<Activations> {
    return this.call({ type: 'activations', planes });
  }
}
