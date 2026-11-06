import { Chess, type Move, type Square } from 'chess.js';
import { BoardView, piecesOf, pieceGlyph, sqIndex, sqName, type Arrow } from './ui/board';
import { MATE } from './engine/search';
import type { Net, ScoredMove } from './engine/net';
import type { Stockfish } from './engine/stockfish';

export interface Timeline {
  startFen: string;
  moves: string[]; // SAN
  ply: number; // currently shown ply (== moves.length while playing)
}

export interface PlayCtx {
  net: Net;
  sf: Stockfish;
  /** Called whenever the position changes (the Explore tab listens to this). */
  onChange: () => void;
}

const START_FEN = new Chess().fen();

type PlayerKind = { kind: 'human' } | { kind: 'deep'; depth: 1 | 2 } | { kind: 'sf'; elo: number };
const PLAYERS: { id: string; label: string; spec: PlayerKind }[] = [
  { id: 'human', label: 'You', spec: { kind: 'human' } },
  { id: 'deep1', label: 'DeepChess (1 move ahead)', spec: { kind: 'deep', depth: 1 } },
  { id: 'deep2', label: 'DeepChess (2 moves ahead)', spec: { kind: 'deep', depth: 2 } },
  ...[1320, 1600, 1900, 2200, 2500, 2800, 3190].map((elo) => ({
    id: `sf${elo}`,
    label: elo === 3190 ? 'Stockfish (full strength)' : `Stockfish (~${elo} Elo)`,
    spec: { kind: 'sf', elo } as PlayerKind,
  })),
];
const spec = (id: string) => PLAYERS.find((p) => p.id === id)!.spec;

export const fmtEval = (s: number) => (Math.abs(s) >= MATE - 1 ? (s > 0 ? '#' : '-#') : (s > 0 ? '+' : '') + s.toFixed(2));
/** pawns -> 0..100 % of the bar that is White's. */
export const barPct = (s: number) => 50 + 50 * Math.tanh(s / 4);

const PIECES = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp', 'bk', 'bq', 'br', 'bb', 'bn', 'bp'];

export function createPlay(root: HTMLElement, ctx: PlayCtx) {
  root.innerHTML = `
  <div class="play">
    <div class="stage">
      <div class="evalbar" id="evalbar"><div class="fill"></div><span class="evtxt"></span></div>
      <div class="boardcol">
        <div class="player-tag top" id="tagTop"></div>
        <div class="boardwrap" id="boardwrap"><div class="promo" id="promo" hidden></div></div>
        <div class="player-tag bottom" id="tagBottom"></div>
        <div class="status" id="status"></div>
        <div class="row actions">
          <button id="flip" title="Flip board">⇅ Flip</button>
          <button id="undo">↶ Undo</button>
          <button id="pause" hidden>⏸ Pause</button>
          <button id="edit">✎ Edit position</button>
        </div>
      </div>
    </div>
    <aside class="side">
      <section class="card">
        <h3>Players</h3>
        <label class="field">White <select id="selW"></select></label>
        <label class="field">Black <select id="selB"></select></label>
        <small class="cap">Pick a bot for both colours to watch them play each other.</small>
        <div class="row buttons">
          <button id="newGame" class="accent">New game</button><button id="swap">Swap sides</button>
        </div>
        <label class="field">Time between bot moves <select id="speed"><option value="0">No wait</option><option value="500" selected>0.5 seconds</option><option value="1000">1 second</option><option value="2000">2 seconds</option></select></label>
      </section>
      <section class="card">
        <h3>What the network thinks</h3>
        <label class="check"><input type="checkbox" id="showEval" checked> Show evaluation</label>
        <label class="check"><input type="checkbox" id="showBest" checked> Show its ranking of every move</label>
        <div class="evalbig" id="evalbig">–</div>
        <small class="cap" id="evalcap">pawns: + is good for White, − is good for Black</small>
        <svg class="spark" id="spark" viewBox="0 0 200 50" preserveAspectRatio="none"></svg>
        <div class="movelist" id="best"></div>
      </section>
      <section class="card" id="editor" hidden>
        <h3>Position editor</h3>
        <div class="palette" id="palette"></div>
        <div class="row">
          <label class="field">To move <select id="edTurn"><option value="w">White</option><option value="b">Black</option></select></label>
        </div>
        <div class="row castle">
          <label class="check"><input type="checkbox" id="cK" checked>O-O</label><label class="check"><input type="checkbox" id="cQ" checked>O-O-O</label>
          <label class="check"><input type="checkbox" id="ck" checked>o-o</label><label class="check"><input type="checkbox" id="cq" checked>o-o-o</label>
        </div>
        <div class="row">
          <button id="edClear">Clear</button><button id="edStart">Start position</button><button id="edCur">Current</button>
        </div>
        <input type="text" id="fen" spellcheck="false" placeholder="FEN">
        <div class="row"><button id="edLoad">Load FEN</button><button id="edApply" class="accent">Play from here</button><button id="edCancel">Cancel</button></div>
        <div class="error" id="edErr"></div>
      </section>
      <section class="card">
        <h3>Moves</h3>
        <ol class="moves" id="moves"></ol>
      </section>
    </aside>
  </div>`;

  const $ = <T extends HTMLElement>(id: string) => root.querySelector<T>('#' + id)!;
  const board = new BoardView($('boardwrap'));
  $('boardwrap').prepend(board.el);

  const selW = $<HTMLSelectElement>('selW'), selB = $<HTMLSelectElement>('selB');
  for (const sel of [selW, selB]) sel.innerHTML = PLAYERS.map((p) => `<option value="${p.id}">${p.label}</option>`).join('');
  selW.value = 'human';
  selB.value = 'deep1';
  if (ctx.net.backend === 'wasm') {
    // 600+ positions per move is too much without a GPU.
    root.querySelectorAll<HTMLOptionElement>('option[value="deep2"]').forEach((o) => (o.textContent += ' — slow without WebGPU'));
  }

  // ---- game state ------------------------------------------------------------------------------
  let startFen = START_FEN;
  let chess = new Chess();
  let hist: Move[] = []; // same as chess.history({ verbose: true }), which replays the whole game on every call
  let evals: number[] = []; // evals[ply] = network eval of the position after `ply` half-moves
  let gen = 0;
  let paused = false;
  let selected: number | null = null;
  let dragFrom: number | null = null;
  let editing = false;
  let scored: ScoredMove[] = [];
  let hover: ScoredMove | null = null;
  let posEval: { score: number; terminal?: string } | null = null;
  let netError = '';

  const sans = () => hist.map((m) => m.san);
  const player = (c: 'w' | 'b'): PlayerKind => spec(c === 'w' ? selW.value : selB.value);
  const bothBots = () => player('w').kind !== 'human' && player('b').kind !== 'human';
  const gameOver = () => chess.isGameOver();

  function status(): string {
    if (chess.isCheckmate()) return `Checkmate — ${chess.turn() === 'w' ? 'Black' : 'White'} wins`;
    if (chess.isStalemate()) return 'Draw by stalemate';
    if (chess.isInsufficientMaterial()) return 'Draw — insufficient material';
    if (chess.isThreefoldRepetition()) return 'Draw by repetition';
    if (chess.isDraw()) return 'Draw (50-move rule)';
    return `${chess.turn() === 'w' ? 'White' : 'Black'} to move${chess.isCheck() ? ' — check' : ''}`;
  }

  // ---- rendering -------------------------------------------------------------------------------
  function render() {
    board.setPieces(piecesOf(chess));
    const last = hist.at(-1);
    const targets = selected !== null && !editing ? chess.moves({ square: sqName(selected) as Square, verbose: true }).map((m) => sqIndex(m.to)) : [];
    const kingSq = chess.isCheck() ? piecesOf(chess).findIndex((p) => p === chess.turn() + 'k') : null;
    board.setHighlights({ selected: editing ? null : selected, targets, last: last ? [sqIndex(last.from), sqIndex(last.to)] : null, check: kingSq });
    drawArrows();

    $('status').textContent = netError || (editing ? 'Editing — click squares to place the selected piece' : status());
    const top = board.flipped ? 'w' : 'b';
    const tag = (c: 'w' | 'b') => `<span class="dot ${c}"></span>${c === 'w' ? 'White' : 'Black'} · ${PLAYERS.find((p) => p.id === (c === 'w' ? selW.value : selB.value))!.label}`;
    $('tagTop').innerHTML = tag(top);
    $('tagBottom').innerHTML = tag(top === 'w' ? 'b' : 'w');
    $('pause').hidden = !bothBots();

    renderEval();
    renderMoves();
  }

  function drawArrows() {
    const arrows: Arrow[] = [];
    if ($<HTMLInputElement>('showBest').checked && scored.length && !editing && !gameOver()) {
      const top = hover ?? scored[0];
      arrows.push({ from: sqIndex(top.move.from), to: sqIndex(top.move.to), color: hover ? '#fbbf24' : '#34d399' });
      if (!hover && scored[1]) arrows.push({ from: sqIndex(scored[1].move.from), to: sqIndex(scored[1].move.to), color: '#60a5fa', opacity: 0.5, width: 0.1 });
    }
    board.setArrows(arrows);
  }

  function renderEval() {
    const show = $<HTMLInputElement>('showEval').checked;
    $('evalbar').style.visibility = show ? 'visible' : 'hidden';
    $('evalbar').classList.toggle('flipped', board.flipped);
    const ev = posEval;
    ($('evalbar').querySelector('.fill') as HTMLElement).style.height = ev ? barPct(ev.score) + '%' : '50%';
    $('evalbig').textContent = !show ? 'Evaluation hidden' : ev ? (ev.terminal === 'draw' ? 'Draw' : fmtEval(ev.score)) : '…';
    $('evalbig').classList.toggle('hidden', !show);
    $('evalcap').hidden = !show;
    $('spark').style.display = show ? '' : 'none';

    // history sparkline
    const pts = evals.map((v, i) => (v === undefined ? null : [(i / Math.max(1, evals.length - 1)) * 200, 25 - 25 * Math.tanh(v / 4)]));
    const path = pts.filter(Boolean).map((p, i) => `${i ? 'L' : 'M'}${p![0].toFixed(1)},${p![1].toFixed(1)}`).join(' ');
    $('spark').innerHTML = show
      ? `<line x1="0" x2="200" y1="25" y2="25" stroke="currentColor" opacity=".25" stroke-dasharray="3 3"/><path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`
      : '';
  }

  /** Only when a new ranking arrives: rebuilding it on hover would re-fire the hover events under the cursor. */
  function renderRanking() {
    const list = $('best');
    const showBest = $<HTMLInputElement>('showBest').checked;
    list.style.display = showBest ? '' : 'none';
    list.classList.remove('stale');
    if (!showBest || editing || gameOver()) return void (list.innerHTML = '');
    const white = chess.turn() === 'w';
    const best = scored[0]?.score ?? 0;
    list.innerHTML = scored
      .map((s, i) => {
        const delta = Math.abs(s.score - best);
        return `<div class="mv${i === 0 ? ' top' : ''}" data-i="${i}"><span class="san">${s.move.san}</span><span class="bar"><i style="width:${Math.max(3, 100 - delta * 18)}%"></i></span><span class="val ${s.score * (white ? 1 : -1) >= 0 ? 'good' : 'bad'}">${s.terminal === 'draw' ? '=' : fmtEval(s.score)}</span></div>`;
      })
      .join('');
    list.scrollTop = 0;
    list.querySelectorAll<HTMLElement>('.mv').forEach((el) => {
      el.onmouseenter = () => ((hover = scored[Number(el.dataset.i)]), drawArrows());
      el.onmouseleave = () => ((hover = null), drawArrows());
    });
  }

  function renderMoves() {
    const h = sans();
    let out = '';
    for (let i = 0; i < h.length; i += 2) out += `<li><b>${i / 2 + 1}.</b><span>${h[i]}</span><span>${h[i + 1] ?? ''}</span></li>`;
    const el = $('moves');
    el.innerHTML = out;
    el.scrollTop = el.scrollHeight; // scrolls only this list, never the page
  }

  // ---- game flow -------------------------------------------------------------------------------
  async function refresh() {
    const my = ++gen;
    const t0 = performance.now();
    hover = null;
    scored = [];
    $('best').classList.add('stale'); // the old numbers stay until the new ones arrive, so nothing jumps
    render();
    ctx.onChange();
    if (editing) return;

    const p = player(chess.turn());
    const showBest = $<HTMLInputElement>('showBest').checked;
    const wantMoves = !gameOver() && (showBest || p.kind === 'deep');
    let res;
    try {
      res = await ctx.net.analyze(startFen, sans(), p.kind === 'deep' ? p.depth : 1, wantMoves);
    } catch (e) {
      if (my === gen) {
        netError = 'The network stopped working: ' + (e as Error).message;
        render();
      }
      return;
    }
    if (!res || my !== gen) return;
    posEval = res;
    evals.length = hist.length + 1;
    evals[hist.length] = res.score;
    scored = res.moves ?? [];
    render();
    renderRanking();
    if (gameOver() || p.kind === 'human' || (paused && bothBots())) return;

    let uci: { from: string; to: string; promotion?: string };
    if (p.kind === 'deep') {
      uci = scored[0].move;
    } else {
      const s = await ctx.sf.bestMove(chess.fen(), p.elo);
      if (my !== gen) return;
      uci = { from: s.slice(0, 2), to: s.slice(2, 4), promotion: s[4] };
    }
    const wait = Number($<HTMLSelectElement>('speed').value) - (performance.now() - t0);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    if (my !== gen || (paused && bothBots())) return;
    play(uci);
  }

  function play(m: { from: string; to: string; promotion?: string }) {
    try {
      hist.push(chess.move(m));
    } catch {
      return;
    }
    selected = null;
    refresh();
  }

  function setPaused(on: boolean) {
    paused = on;
    $('pause').textContent = on ? '▶ Resume' : '⏸ Pause';
  }

  /** Look at the game from the side the person is playing. */
  function faceHuman() {
    const w = player('w').kind === 'human', b = player('b').kind === 'human';
    if (w !== b) board.setOrientation(w ? 'w' : 'b');
  }

  function newGame(fen = START_FEN) {
    startFen = fen;
    chess = new Chess(fen);
    hist = [];
    evals = [];
    selected = null;
    posEval = null;
    setPaused(false);
    faceHuman();
    editing = false;
    $('editor').hidden = true;
    $('edit').textContent = '✎ Edit position';
    refresh();
  }

  function setEditing(on: boolean) {
    editing = on;
    $('editor').hidden = !on;
    $('edit').textContent = on ? '✕ Close editor' : '✎ Edit position';
    if (on) {
      gen++; // stop any bot that is thinking
      board.setArrows([]);
      loadEditorFrom(chess.fen());
      $('status').textContent = 'Editing — click squares to place the selected piece';
    } else refresh();
  }

  // ---- human input -----------------------------------------------------------------------------
  const humanTurn = () => !editing && !gameOver() && player(chess.turn()).kind === 'human';

  function tryMove(from: number, to: number) {
    const cands = chess.moves({ square: sqName(from) as Square, verbose: true }).filter((m) => m.to === sqName(to));
    if (!cands.length) return;
    if (cands[0].promotion) return askPromotion(from, to, cands[0].color);
    play({ from: sqName(from), to: sqName(to) });
  }

  function askPromotion(from: number, to: number, color: string) {
    const el = $('promo');
    el.hidden = false;
    el.innerHTML = ['q', 'r', 'b', 'n'].map((t) => `<button data-t="${t}" class="${color === 'w' ? 'wp' : 'bp'}">${pieceGlyph(t)}</button>`).join('');
    el.querySelectorAll<HTMLElement>('button').forEach(
      (b) =>
        (b.onclick = () => {
          el.hidden = true;
          play({ from: sqName(from), to: sqName(to), promotion: b.dataset.t });
        }),
    );
  }

  board.onDown = (i) => {
    if (editing) return editorClick(i);
    $('promo').hidden = true;
    if (!humanTurn()) return;
    const targets = selected !== null ? chess.moves({ square: sqName(selected) as Square, verbose: true }).map((m) => sqIndex(m.to)) : [];
    if (selected !== null && targets.includes(i)) {
      tryMove(selected, i);
      dragFrom = null;
      return;
    }
    const p = chess.get(sqName(i) as Square);
    selected = p && p.color === chess.turn() ? i : null;
    dragFrom = selected;
    render();
  };
  board.onUp = (i) => {
    if (editing || dragFrom === null) return;
    const from = dragFrom;
    dragFrom = null;
    if (i !== from && chess.moves({ square: sqName(from) as Square, verbose: true }).some((m) => m.to === sqName(i))) tryMove(from, i);
  };

  // ---- editor ----------------------------------------------------------------------------------
  let edBoard: (string | null)[] = Array(64).fill(null);
  let tool: string | null = 'wp'; // null = eraser

  $('palette').innerHTML =
    PIECES.map((p) => `<button data-p="${p}" class="${p[0] === 'w' ? 'wp' : 'bp'}">${pieceGlyph(p[1])}</button>`).join('') + '<button data-p="" title="Eraser">⌫</button>';
  const paletteBtns = () => root.querySelectorAll<HTMLElement>('#palette button');
  function markTool() {
    paletteBtns().forEach((b) => b.classList.toggle('on', (b.dataset.p || null) === tool));
  }
  paletteBtns().forEach((b) => (b.onclick = () => ((tool = b.dataset.p || null), markTool())));
  markTool();

  function editorFen(): string {
    const rows: string[] = [];
    for (let r = 0; r < 8; r++) {
      let s = '', empty = 0;
      for (let c = 0; c < 8; c++) {
        const p = edBoard[r * 8 + c];
        if (!p) empty++;
        else {
          if (empty) (s += empty), (empty = 0);
          s += p[0] === 'w' ? p[1].toUpperCase() : p[1];
        }
      }
      rows.push(s + (empty || ''));
    }
    const at = (name: string) => edBoard[sqIndex(name)];
    const rights = [
      ['cK', 'K', at('e1') === 'wk' && at('h1') === 'wr'],
      ['cQ', 'Q', at('e1') === 'wk' && at('a1') === 'wr'],
      ['ck', 'k', at('e8') === 'bk' && at('h8') === 'br'],
      ['cq', 'q', at('e8') === 'bk' && at('a8') === 'br'],
    ] as const;
    let castle = '';
    for (const [id, ch, ok] of rights) {
      const box = $<HTMLInputElement>(id);
      box.disabled = !ok;
      if (ok && box.checked) castle += ch;
    }
    return `${rows.join('/')} ${$<HTMLSelectElement>('edTurn').value} ${castle || '-'} - 0 1`;
  }

  function editorSync() {
    const fen = editorFen();
    $<HTMLInputElement>('fen').value = fen;
    board.setPieces(edBoard);
    board.setHighlights({});
    board.setArrows([]);
    $('edErr').textContent = validate(fen) ?? '';
  }

  function editorClick(i: number) {
    edBoard[i] = tool && edBoard[i] === tool ? null : tool;
    editorSync();
  }

  function loadEditorFrom(fen: string) {
    const c = new Chess(fen);
    edBoard = piecesOf(c);
    $<HTMLSelectElement>('edTurn').value = c.turn();
    const cr = fen.split(' ')[2];
    for (const [id, ch] of [['cK', 'K'], ['cQ', 'Q'], ['ck', 'k'], ['cq', 'q']]) $<HTMLInputElement>(id).checked = cr.includes(ch);
    editorSync();
  }

  /** Returns an error message, or null if the network can be shown this position. */
  function validate(fen: string): string | null {
    const placement = fen.split(' ')[0];
    const count = (ch: string) => [...placement].filter((c) => c === ch).length;
    if (count('K') !== 1 || count('k') !== 1) return 'Each side needs exactly one king.';
    const rows = placement.split('/');
    if (/[Pp]/.test(rows[0] + rows[7])) return 'Pawns cannot stand on the first or last rank.';
    if (count('P') > 8 || count('p') > 8) return 'Too many pawns.';
    try {
      new Chess(fen);
    } catch (e) {
      return String((e as Error).message).replace(/^Invalid FEN: ?/, '');
    }
    // The side that just moved must not have left its own king in check.
    const parts = fen.split(' ');
    parts[1] = parts[1] === 'w' ? 'b' : 'w';
    parts[2] = '-';
    try {
      if (new Chess(parts.join(' ')).isCheck()) return 'The side not to move is in check — that position is illegal.';
    } catch {
      return null;
    }
    return null;
  }

  for (const id of ['edTurn', 'cK', 'cQ', 'ck', 'cq']) $(id).addEventListener('change', editorSync);
  $('edClear').onclick = () => ((edBoard = Array(64).fill(null)), editorSync());
  $('edStart').onclick = () => loadEditorFrom(START_FEN);
  $('edCur').onclick = () => loadEditorFrom(chess.fen());
  $('edLoad').onclick = () => {
    try {
      loadEditorFrom($<HTMLInputElement>('fen').value.trim());
    } catch (e) {
      $('edErr').textContent = String((e as Error).message);
    }
  };
  $('edApply').onclick = () => {
    const fen = editorFen();
    const err = validate(fen);
    if (err) return void ($('edErr').textContent = err);
    newGame(fen);
  };
  $('edCancel').onclick = () => setEditing(false);
  $('edit').onclick = () => setEditing(!editing);

  // ---- controls --------------------------------------------------------------------------------
  $('newGame').onclick = () => newGame();
  $('swap').onclick = () => {
    [selW.value, selB.value] = [selB.value, selW.value];
    newGame();
  };
  for (const sel of [selW, selB]) {
    sel.onchange = () => {
      setPaused(false);
      faceHuman();
      refresh();
    };
  }
  $('flip').onclick = () => (board.setOrientation(board.flipped ? 'w' : 'b'), editing ? editorSync() : render());
  $('undo').onclick = () => {
    if (editing || !hist.length) return;
    const anyHuman = !bothBots();
    do {
      chess.undo();
      hist.pop();
    } while (anyHuman && hist.length && player(chess.turn()).kind !== 'human');
    selected = null;
    refresh();
  };
  $('pause').onclick = () => {
    setPaused(!paused);
    if (!paused) refresh();
  };
  $('showEval').addEventListener('change', renderEval);
  $('showBest').addEventListener('change', () => refresh());

  newGame();

  return {
    timeline: (): Timeline => ({ startFen, moves: sans(), ply: hist.length }),
    /** Game position after `ply` half-moves, with history intact (the encoder needs the last move). */
    chessAt(ply: number): Chess {
      const c = new Chess(startFen);
      for (const m of hist.slice(0, ply)) c.move(m.san);
      return c;
    },
  };
}

export type Play = ReturnType<typeof createPlay>;
