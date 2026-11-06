import type { Chess } from 'chess.js';

/** Plane layout (matches scripts/infer_and_train.py board_to_array): 19 planes of 8x8, row 0 = rank 8. */
export const PLANES = 19;
export const PLANE_SIZE = PLANES * 64;
export const PLANE_NAMES = [
  'White pawns', 'White knights', 'White bishops', 'White rooks', 'White queens', 'White king',
  'Black pawns', 'Black knights', 'Black bishops', 'Black rooks', 'Black queens', 'Black king',
  'White O-O', 'White O-O-O', 'Black O-O', 'Black O-O-O',
  'En passant square', 'White to move', 'Constant 1',
];
const PIECE_INDEX: Record<string, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

/**
 * The training data came from python-chess, whose ep_square is set after EVERY double pawn push,
 * while chess.js only reports it when a capture is legal. So derive it from the move that was played.
 */
export function epAfterMove(m: { flags: string; to: string; color: string }): string | null {
  return m.flags.includes('b') ? `${m.to[0]}${m.color === 'w' ? 3 : 6}` : null;
}

/** En-passant square when the caller does not know the last move. Replays the game, so it is slow. */
function epOfPosition(chess: Chess): string | null {
  const last = chess.history({ verbose: true }).at(-1);
  if (last) return epAfterMove(last);
  const ep = chess.fen().split(' ')[3];
  return ep && ep !== '-' ? ep : null;
}

/** Fill `out` (zeroed, at `offset`) with the network input. Pass `ep` when known: it avoids replaying the game. */
export function encodeInto(chess: Chess, out: Float32Array, offset = 0, ep?: string | null): void {
  const board = chess.board();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p) out[offset + ((p.color === 'w' ? 0 : 6) + PIECE_INDEX[p.type]) * 64 + r * 8 + c] = 1;
    }
  }
  const w = chess.getCastlingRights('w'), b = chess.getCastlingRights('b');
  [w.k, w.q, b.k, b.q].forEach((on, i) => on && out.fill(1, offset + (12 + i) * 64, offset + (13 + i) * 64));
  const square = ep === undefined ? epOfPosition(chess) : ep;
  if (square) out[offset + 16 * 64 + (8 - Number(square[1])) * 8 + (square.charCodeAt(0) - 97)] = 1;
  if (chess.turn() === 'w') out.fill(1, offset + 17 * 64, offset + 18 * 64);
  out.fill(1, offset + 18 * 64, offset + 19 * 64);
}

export function encode(chess: Chess): Float32Array {
  const out = new Float32Array(PLANE_SIZE);
  encodeInto(chess, out);
  return out;
}
