import type { Chess } from 'chess.js';
import { PLANE_SIZE, encodeInto, epAfterMove } from './encode';
import type { Analysis, ScoredMove } from './protocol';

export const MATE = 100;

export type Evaluate = (planes: Float32Array, n: number) => Promise<Float32Array>;

/** Terminal value (White's view) if the game is over in `chess`, otherwise null. */
export function terminalScore(chess: Chess): number | null {
  if (chess.isCheckmate()) return chess.turn() === 'w' ? -MATE : MATE;
  if (chess.isDraw()) return 0;
  return null;
}

type Leaf = { terminal?: number; index?: number };

/**
 * Evaluate the position and, if asked, every legal move by running the network on the resulting positions.
 * depth 1 evaluates each child; depth 2 also considers every reply (full-width minimax).
 * All positions go through the network together in batches.
 */
export async function analyze(evaluate: Evaluate, chess: Chess, depth: 1 | 2, wantMoves: boolean): Promise<Analysis> {
  const rootTerminal = terminalScore(chess);
  if (rootTerminal !== null) return { score: rootTerminal, terminal: rootTerminal === 0 ? 'draw' : 'mate', moves: [] };

  const buf: Float32Array[] = [];
  const queue = (ep: string | null | undefined): Leaf => {
    const a = new Float32Array(PLANE_SIZE);
    encodeInto(chess, a, 0, ep);
    buf.push(a);
    return { index: buf.length - 1 };
  };
  const root = queue(undefined);

  const legal = wantMoves ? chess.moves({ verbose: true }) : [];
  const white = chess.turn() === 'w';
  const nodes = legal.map((m) => {
    chess.move(m);
    const t = terminalScore(chess);
    let own: Leaf = { terminal: t ?? undefined };
    let replies: Leaf[] | undefined;
    if (t === null) {
      if (depth === 1) own = queue(epAfterMove(m));
      else {
        replies = chess.moves({ verbose: true }).map((r) => {
          chess.move(r);
          const rt = terminalScore(chess);
          const leaf = rt !== null ? { terminal: rt } : queue(epAfterMove(r));
          chess.undo();
          return leaf;
        });
      }
    }
    const blackToMove = chess.turn() === 'b';
    chess.undo();
    return { own, replies, blackToMove };
  });

  const flat = new Float32Array(buf.length * PLANE_SIZE);
  buf.forEach((a, i) => flat.set(a, i * PLANE_SIZE));
  const evals = await evaluate(flat, buf.length);
  const val = (l: Leaf) => (l.terminal !== undefined ? l.terminal : evals[l.index!]);

  const moves = legal.map((m, i): ScoredMove => {
    const n = nodes[i];
    const score = n.replies ? (n.blackToMove ? Math.min(...n.replies.map(val)) : Math.max(...n.replies.map(val))) : val(n.own);
    const terminal = n.own.terminal === undefined ? undefined : n.own.terminal === 0 ? 'draw' : 'mate';
    return { move: { san: m.san, from: m.from, to: m.to, promotion: m.promotion }, score, terminal };
  });
  moves.sort((a, b) => (white ? b.score - a.score : a.score - b.score));
  return { score: val(root), moves: wantMoves ? moves : undefined };
}
