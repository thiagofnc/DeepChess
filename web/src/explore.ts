import { Chess } from 'chess.js';
import { BoardView, divergingColor, heatColor, piecesOf, sqName } from './ui/board';
import { PLANE_NAMES, PLANE_SIZE, encode } from './engine/encode';
import { STAGES, type Activations, type Net, type Stage } from './engine/net';
import { terminalScore } from './engine/search';
import { fmtEval, type Play } from './play';

type Key = 'planes' | Stage;
const ORDER: Key[] = ['planes', ...STAGES];

const LABEL: Record<Key, string> = {
  planes: 'Input', stem: 'Stem', block0: 'Block 1', block1: 'Block 2', block2: 'Block 3', block3: 'Block 4',
  block4: 'Block 5', block5: 'Block 6', block6: 'Block 7', head: 'Head', fc1: 'Hidden', eval: 'Output',
};
const CHANNELS: Partial<Record<Key, number>> = { planes: 19, stem: 128, head: 32 };
for (let i = 0; i < 7; i++) CHANNELS[`block${i}` as Key] = 128;

const INFO: Record<Key, string> = {
  planes: 'What the network is actually given: 19 binary 8×8 planes — one per piece type and colour, four castling rights, the en-passant square, and side to move. It never sees a "move", only this snapshot.',
  stem: 'A 3×3 convolution turns the 19 input planes into 128 feature maps. Each channel is a simple local pattern detector: "white pawn with a friend diagonally behind it", "attacked-looking square", and so on.',
  block0: 'Residual block 1. Two 3×3 convolutions whose result is added back onto the previous maps — the block only has to learn a correction.',
  block1: 'Residual block 2. Receptive field grows by 4 squares per block, so features start to combine pieces further apart.',
  block2: 'Residual block 3. Patterns here tend to be about structures: pawn chains, king shelter, open files.',
  block3: 'Residual block 4. By now every square can "see" the entire board (a 3×3 conv stack reaches ±7 squares after four blocks).',
  block4: 'Residual block 5. Features are increasingly abstract and tied to the evaluation rather than to piece placement.',
  block5: 'Residual block 6. Late blocks mostly refine what earlier blocks already decided.',
  block6: 'Residual block 7 — the last convolutional features: a 128-channel summary of the position, still spatial.',
  head: 'A 1×1 convolution squeezes 128 channels to 32. These 32 maps (2048 numbers) are everything the final layers get to see.',
  fc1: 'The 2048 head numbers are flattened and mixed by a fully-connected layer into 256 hidden units (GELU). The evaluation is a weighted sum of these 256 numbers.',
  eval: 'The evaluation in pawns from White\'s point of view: Σ(hidden unit × output weight) + bias, scaled ×10.',
};

type HeatMode = 'mean' | 'max' | 'channel' | 'occlusion';

export function createExplore(root: HTMLElement, net: Net, play: Play) {
  root.innerHTML = `
  <div class="explore">
    <div class="ex-left">
      <section class="card">
        <div class="timeline"><label>Position <input type="range" id="exPly" min="0" max="0" value="0"></label><span id="exPlyTxt"></span></div>
        <div class="ex-headline"><span id="exEval" class="evalbig small">–</span><small id="exFen"></small></div>
        <div class="boardwrap" id="exBoardWrap"></div>
        <div class="seg" id="exMode">
          <button data-m="mean">Mean</button><button data-m="max">Max</button><button data-m="channel">Channel</button><button data-m="occlusion">Piece impact</button>
        </div>
        <p class="hint" id="exHint"></p>
      </section>
    </div>
    <div class="ex-right">
      <section class="card">
        <h3>The network, layer by layer <small>(click a layer — squares show the average activation across its channels)</small></h3>
        <div class="ribbon" id="ribbon"></div>
      </section>
      <section class="card" id="detail"></section>
      <section class="card">
        <h3>How much does each block change things?</h3>
        <div id="deltas"></div>
      </section>
      <section class="card">
        <h3>Piece impact (occlusion)</h3>
        <p class="hint">Remove one piece at a time from the input and ask the network again. Green: the piece pushes the evaluation toward White; red: toward Black. This is the quickest way to see what the net thinks matters.</p>
        <div class="row"><button id="occ" class="accent">Compute piece impact</button><span id="occTxt" class="hint"></span></div>
      </section>
    </div>
  </div>`;

  const $ = <T extends HTMLElement>(id: string) => root.querySelector<T>('#' + id)!;
  const board = new BoardView($('exBoardWrap'));

  let fc2: { weights: number[]; bias: number } | null = null;
  fetch(new URL('model/fc2.json', document.baseURI)).then((r) => r.json()).then((j) => ((fc2 = j), act && renderDetail()));

  let ply = 0;
  let chess = new Chess();
  let act: Activations | null = null;
  let stage: Key = 'block3';
  let mode: HeatMode = 'mean';
  let channel = 0;
  let sortChannels = true;
  let fc1View: 'contribution' | 'activation' = 'contribution';
  let occlusion: Float32Array | null = null; // per square: how much the piece there moves eval toward White
  let token = 0;

  // ---- data ------------------------------------------------------------------------------------
  const tensor = (k: Key): Float32Array => (k === 'planes' ? act!.planes : act![k]);

  /** 64 values summarising a stage across channels (or a single channel). */
  function spatial(k: Key, m: 'mean' | 'max' | 'channel', ch = 0): Float32Array {
    const C = CHANNELS[k]!;
    const t = tensor(k);
    const out = new Float32Array(64);
    if (m === 'channel') return out.set(t.subarray(ch * 64, ch * 64 + 64)), out;
    for (let s = 0; s < 64; s++) {
      let acc = m === 'max' ? -Infinity : 0;
      for (let c = 0; c < C; c++) {
        const v = t[c * 64 + s];
        acc = m === 'max' ? Math.max(acc, v) : acc + v;
      }
      out[s] = m === 'max' ? acc : acc / C;
    }
    return out;
  }

  const maxOf = (a: ArrayLike<number>) => {
    let m = 1e-9;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
    return m;
  };

  async function load() {
    const my = ++token;
    chess = play.chessAt(ply);
    const planes = encode(chess);
    act = await net.activations(planes);
    if (my !== token) return;
    occlusion = null;
    $('occTxt').textContent = '';
    renderAll();
  }

  // ---- rendering -------------------------------------------------------------------------------
  function paintBoard() {
    board.setPieces(piecesOf(chess));
    board.setHighlights({});
    board.setArrows([]);
    board.setLabels(null);
    if (!act) return board.setHeat(null, heatColor);
    if (mode === 'occlusion') {
      if (!occlusion) {
        board.setHeat(null, heatColor);
        $('exHint').textContent = 'Press “Compute piece impact” on the right to fill this in.';
        return;
      }
      const m = maxOf(occlusion);
      board.setHeat(occlusion, (v) => (v === 0 ? 'transparent' : divergingColor(v / m)));
      board.setLabels(Array.from(occlusion, (v) => (v === 0 ? null : (v > 0 ? '+' : '') + v.toFixed(2))));
      $('exHint').textContent = 'Each number is how much the evaluation changes in White\'s favour because that piece is on the board.';
      return;
    }
    const k = stage;
    if (!CHANNELS[k]) {
      board.setHeat(null, heatColor);
      $('exHint').textContent = 'This layer is no longer spatial — it is a flat list of numbers, so there is nothing to draw on the board.';
      return;
    }
    const m = mode === 'channel' ? 'channel' : (mode as 'mean' | 'max');
    const v = spatial(k, m, channel);
    const mx = maxOf(v);
    board.setHeat(v, (x) => (x <= 0 ? 'rgba(0,0,0,.0)' : heatColor(x / mx)));
    $('exHint').textContent =
      mode === 'channel'
        ? `${LABEL[k]} · channel ${channel}${k === 'planes' ? ' — ' + PLANE_NAMES[channel] : ''}: bright squares are where this feature fires.`
        : `${LABEL[k]} · ${mode} over ${CHANNELS[k]} channels: brighter squares carry more activation.`;
  }

  function tile(canvas: HTMLCanvasElement, v: ArrayLike<number>, mx: number, px = 6) {
    canvas.width = canvas.height = 8 * px;
    const g = canvas.getContext('2d')!;
    for (let s = 0; s < 64; s++) {
      g.fillStyle = heatColor(Math.max(0, v[s]) / mx, 1);
      g.fillRect((s % 8) * px, Math.floor(s / 8) * px, px, px);
    }
  }

  function renderRibbon() {
    const el = $('ribbon');
    el.innerHTML = '';
    for (const k of ORDER) {
      const d = document.createElement('button');
      d.className = 'rtile' + (k === stage ? ' on' : '');
      const c = document.createElement('canvas');
      c.className = 'mini';
      d.appendChild(c);
      d.insertAdjacentHTML('beforeend', `<span>${LABEL[k]}</span>`);
      if (act) {
        if (CHANNELS[k]) {
          const v = spatial(k, 'mean');
          tile(c, v, maxOf(v), 5);
        } else if (k === 'eval') {
          c.replaceWith(Object.assign(document.createElement('div'), { className: 'mini big', textContent: fmtEval(act.eval[0]) }));
        } else {
          c.replaceWith(Object.assign(document.createElement('div'), { className: 'mini', textContent: '256' }));
        }
      }
      d.onclick = () => {
        stage = k;
        if (mode === 'occlusion') mode = 'mean';
        if (mode === 'channel' && !CHANNELS[k]) mode = 'mean';
        channel = Math.min(channel, (CHANNELS[k] ?? 1) - 1);
        renderAll();
      };
      el.appendChild(d);
    }
  }

  function renderDetail() {
    const el = $('detail');
    if (!act) return void (el.innerHTML = '<p class="hint">Running the network…</p>');
    const k = stage;
    const C = CHANNELS[k];
    let html = `<h3>${LABEL[k]}</h3><p class="hint">${INFO[k]}</p>`;
    if (C) {
      const t = tensor(k);
      const stats = Array.from({ length: C }, (_, c) => {
        let sum = 0, mx = 0, nz = 0;
        for (let s = 0; s < 64; s++) {
          const v = t[c * 64 + s];
          sum += v;
          mx = Math.max(mx, v);
          if (v > 1e-6) nz++;
        }
        return { c, sum, mx, nz };
      });
      const order = stats.map((s) => s.c);
      if (sortChannels && k !== 'planes') order.sort((a, b) => stats[b].sum - stats[a].sum);
      const dead = stats.filter((s) => s.mx <= 1e-6).length;
      html += `<div class="row"><label class="check"><input type="checkbox" id="sortCh" ${sortChannels ? 'checked' : ''}> Sort by activity</label>
        <small>${C - dead} of ${C} channels active · click one to see it on the board</small></div><div class="chgrid" id="chgrid"></div>`;
      el.innerHTML = html;
      const grid = el.querySelector('#chgrid')!;
      for (const c of order) {
        const b = document.createElement('button');
        b.className = 'ch' + (mode === 'channel' && c === channel ? ' on' : '') + (stats[c].mx <= 1e-6 ? ' dead' : '');
        const cv = document.createElement('canvas');
        tile(cv, t.subarray(c * 64, c * 64 + 64), Math.max(stats[c].mx, 1e-9), 5);
        b.appendChild(cv);
        b.insertAdjacentHTML('beforeend', `<span>${k === 'planes' ? PLANE_NAMES[c] : c}</span>`);
        b.title = `channel ${c}: max ${stats[c].mx.toFixed(2)}, ${stats[c].nz}/64 squares active`;
        b.onclick = () => {
          channel = c;
          mode = 'channel';
          renderAll();
        };
        grid.appendChild(b);
      }
      el.querySelector<HTMLInputElement>('#sortCh')!.onchange = (e) => ((sortChannels = (e.target as HTMLInputElement).checked), renderDetail());
    } else if (k === 'fc1') {
      html += renderFc1();
      el.innerHTML = html;
      el.querySelectorAll<HTMLElement>('[data-v]').forEach((b) => (b.onclick = () => ((fc1View = b.dataset.v as any), renderDetail())));
    } else {
      const total = act.eval[0];
      html += `<div class="evalbig">${fmtEval(total)}</div><p class="hint">pawns, White's view${terminalScore(chess) !== null ? ' — note the game is over here; the network itself never saw mates' : ''}.</p>`;
      el.innerHTML = html;
    }
  }

  function renderFc1(): string {
    const a = act!.fc1;
    const w = fc2?.weights;
    const items = Array.from(a, (v, i) => ({ i, v, c: w ? v * w[i] : 0 }));
    const key = fc1View === 'contribution' && w ? 'c' : 'v';
    const sorted = [...items].sort((x, y) => y[key] - x[key]);
    const mx = maxOf(sorted.map((s) => s[key]));
    const W = 512, H = 120, bw = W / sorted.length;
    const bars = sorted
      .map((s, n) => {
        const h = (Math.abs(s[key]) / mx) * (H / 2 - 4);
        const y = s[key] >= 0 ? H / 2 - h : H / 2;
        return `<rect x="${(n * bw).toFixed(2)}" y="${y.toFixed(1)}" width="${(bw * 0.8).toFixed(2)}" height="${Math.max(0.5, h).toFixed(1)}" fill="${s[key] >= 0 ? '#34d399' : '#f85656'}"><title>unit ${s.i}: activation ${s.v.toFixed(2)}${w ? `, contributes ${s.c >= 0 ? '+' : ''}${s.c.toFixed(2)} pawns` : ''}</title></rect>`;
      })
      .join('');
    const sum = w ? items.reduce((p, q) => p + q.c, 0) : 0;
    return `<div class="seg"><button data-v="contribution" class="${key === 'c' ? 'on' : ''}">Contribution to eval</button><button data-v="activation" class="${key === 'v' ? 'on' : ''}">Activation</button></div>
      <svg class="bars" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="currentColor" opacity=".3"/>${bars}</svg>
      <p class="hint">${w ? `256 hidden units sorted left→right. Their weighted sum is ${sum >= 0 ? '+' : ''}${sum.toFixed(2)}, plus a bias of ${fc2!.bias.toFixed(2)} = ${(sum + fc2!.bias).toFixed(2)} pawns (matches the output).` : 'Loading output weights…'}</p>`;
  }

  function renderDeltas() {
    if (!act) return void ($('deltas').innerHTML = '');
    const keys = ['stem', ...Array.from({ length: 7 }, (_, i) => `block${i}`)] as Stage[];
    const rows = keys.map((k, n) => {
      const cur = act![k];
      const prev = n === 0 ? null : act![keys[n - 1]];
      let d = 0, mag = 0;
      for (let i = 0; i < cur.length; i++) {
        d += Math.abs(cur[i] - (prev ? prev[i] : 0));
        mag += Math.abs(cur[i]);
      }
      return { k, d: d / cur.length, mag: mag / cur.length };
    });
    const mx = Math.max(...rows.map((r) => r.d));
    $('deltas').innerHTML =
      `<p class="hint">Average size of the change each block makes to the board representation (|output − input|). Big bars = that block is doing real work on this position.</p>` +
      rows
        .map((r) => `<div class="drow"><span>${LABEL[r.k]}</span><span class="bar"><i style="width:${(r.d / mx) * 100}%"></i></span><span class="val">${r.d.toFixed(3)}</span></div>`)
        .join('');
  }

  function renderAll() {
    $('exEval').textContent = act ? (terminalScore(chess) !== null ? 'Game over' : fmtEval(act.eval[0])) : '…';
    $('exFen').textContent = chess.fen();
    root.querySelectorAll<HTMLElement>('#exMode button').forEach((b) => b.classList.toggle('on', b.dataset.m === mode));
    root.querySelectorAll<HTMLButtonElement>('#exMode button').forEach((b) => (b.disabled = b.dataset.m === 'channel' && !CHANNELS[stage]));
    paintBoard();
    renderRibbon();
    renderDetail();
    renderDeltas();
  }

  // ---- controls --------------------------------------------------------------------------------
  root.querySelectorAll<HTMLElement>('#exMode button').forEach(
    (b) =>
      (b.onclick = () => {
        mode = b.dataset.m as HeatMode;
        if (mode !== 'occlusion' && !CHANNELS[stage]) stage = 'block3';
        renderAll();
      }),
  );
  const slider = $<HTMLInputElement>('exPly');
  slider.oninput = () => {
    ply = Number(slider.value);
    syncPlyText();
    load();
  };
  function syncPlyText() {
    const t = play.timeline();
    $('exPlyTxt').textContent = ply === 0 ? 'start position' : `after ${Math.ceil(ply / 2)}${ply % 2 ? '.' : '…'} ${t.moves[ply - 1]}`;
  }

  $('occ').onclick = async () => {
    if (!act) return;
    const planes = act.planes;
    const items: number[] = [];
    for (let s = 0; s < 64; s++) {
      for (let p = 0; p < 12; p++) if (planes[p * 64 + s] && p % 6 !== 5) items.push(p * 64 + s);
    }
    const batch = new Float32Array((items.length + 1) * PLANE_SIZE);
    batch.set(planes, 0);
    items.forEach((idx, n) => {
      batch.set(planes, (n + 1) * PLANE_SIZE);
      batch[(n + 1) * PLANE_SIZE + idx] = 0;
    });
    $('occTxt').textContent = 'Running…';
    const ev = await net.evaluate(batch, items.length + 1);
    const out = new Float32Array(64);
    items.forEach((idx, n) => (out[idx % 64] = ev[0] - ev[n + 1]));
    occlusion = out;
    mode = 'occlusion';
    $('occTxt').textContent = `${items.length} pieces re-evaluated (kings are skipped).`;
    renderAll();
  };

  void sqName;
  return {
    /** Call when the tab becomes visible or the game changed. */
    sync() {
      const n = play.timeline().moves.length;
      slider.max = String(n);
      ply = n;
      slider.value = String(n);
      syncPlyText();
      load();
    },
  };
}
