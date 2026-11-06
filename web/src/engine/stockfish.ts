/** Thin UCI wrapper around the single-threaded Stockfish lite build (runs in its own Web Worker, loaded lazily). */
export class Stockfish {
  private worker: Worker | null = null;
  private listeners = new Set<(line: string) => void>();
  private ready: Promise<void> | null = null;
  private busy: Promise<unknown> = Promise.resolve();

  private send(cmd: string) {
    this.worker!.postMessage(cmd);
  }

  private waitFor(match: (line: string) => boolean): Promise<string> {
    return new Promise((resolve) => {
      const fn = (line: string) => {
        if (match(line)) {
          this.listeners.delete(fn);
          resolve(line);
        }
      };
      this.listeners.add(fn);
    });
  }

  init(): Promise<void> {
    this.ready ??= (async () => {
      this.worker = new Worker(new URL('vendor/stockfish/stockfish-19-lite-single.js', document.baseURI));
      this.worker.onmessage = (e) => {
        const line = String(e.data);
        for (const l of [...this.listeners]) l(line);
      };
      const uciok = this.waitFor((l) => l === 'uciok');
      this.send('uci');
      await uciok;
      const readyok = this.waitFor((l) => l === 'readyok');
      this.send('isready');
      await readyok;
    })();
    return this.ready;
  }

  /** Best move in UCI notation ("e2e4", "e7e8q") for `fen`, playing at roughly the given Elo (1320-3190). */
  async bestMove(fen: string, elo: number): Promise<string> {
    await this.init();
    this.stop(); // abandon any search still running for a position we no longer care about
    const run = this.busy.then(() => this.search(fen, elo));
    this.busy = run.catch(() => {});
    return run;
  }

  private async search(fen: string, elo: number): Promise<string> {
    const limited = elo < 3190;
    this.send(`setoption name UCI_LimitStrength value ${limited}`);
    if (limited) this.send(`setoption name UCI_Elo value ${elo}`);
    this.send(`position fen ${fen}`);
    const done = this.waitFor((l) => l.startsWith('bestmove'));
    this.send(`go movetime ${limited ? 400 : 1000}`);
    return (await done).split(' ')[1];
  }

  stop() {
    this.worker?.postMessage('stop');
  }
}
