/** Messages between the page and the network worker. */
export const STAGES = ['stem', 'block0', 'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 'head', 'fc1', 'eval'] as const;
export type Stage = (typeof STAGES)[number];
export type Activations = Record<Stage, Float32Array> & { planes: Float32Array };
export type Backend = 'webgpu' | 'wasm';

export interface ScoredMove {
  move: { san: string; from: string; to: string; promotion?: string };
  /** Pawns from White's point of view (±MATE for a checkmate on the board). */
  score: number;
  /** Terminal result after this move, if any. */
  terminal?: 'mate' | 'draw';
}

export interface Analysis {
  /** Evaluation of the position itself. */
  score: number;
  terminal?: 'mate' | 'draw';
  /** Every legal move, best for the side to move first (when requested). */
  moves?: ScoredMove[];
}

export type Request =
  | { id: number; type: 'init'; base: string; backends: Backend[] }
  | { id: number; type: 'analyze'; startFen: string; sans: string[]; depth: 1 | 2; wantMoves: boolean }
  | { id: number; type: 'evaluate'; planes: Float32Array; n: number }
  | { id: number; type: 'activations'; planes: Float32Array };

export type Response =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string }
  | { id: -1; status: string };
