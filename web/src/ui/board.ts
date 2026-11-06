import type { Chess } from 'chess.js';

/** Board index: 0 = a8 ... 63 = h1 (row-major from White's view), same as the network's 8x8 planes. */
export const sqName = (i: number) => `${'abcdefgh'[i % 8]}${8 - Math.floor(i / 8)}`;
export const sqIndex = (name: string) => (8 - Number(name[1])) * 8 + (name.charCodeAt(0) - 97);

const GLYPH: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
export const pieceGlyph = (type: string) => GLYPH[type] + '︎';

/** 64 entries like 'wp' / 'bk' / null for the position in `chess`. */
export function piecesOf(chess: Chess): (string | null)[] {
  return chess.board().flat().map((p) => (p ? p.color + p.type : null));
}

export interface Arrow {
  from: number;
  to: number;
  color: string;
  width?: number;
  opacity?: number;
}

export interface Highlights {
  selected?: number | null;
  targets?: Iterable<number>;
  last?: [number, number] | null;
  check?: number | null;
}

export type ColorFn = (v: number) => string;

const SVG = 'http://www.w3.org/2000/svg';

export class BoardView {
  readonly el = document.createElement('div');
  private cells: HTMLElement[] = [];
  private shown: (string | null)[] = Array(64).fill(null);
  private svg = document.createElementNS(SVG, 'svg');
  private orientation: 'w' | 'b' = 'w';
  onDown: (idx: number) => void = () => {};
  onUp: (idx: number) => void = () => {};

  constructor(parent: HTMLElement) {
    this.el.className = 'board';
    for (let i = 0; i < 64; i++) {
      const c = document.createElement('div');
      c.className = ((i % 8) + Math.floor(i / 8)) % 2 === 0 ? 'sq light' : 'sq dark';
      c.innerHTML = '<i class="heat"></i><span class="piece"></span><b class="coord-r"></b><b class="coord-f"></b><em class="lbl"></em><u class="dot"></u>';
      this.cells.push(c);
      this.el.appendChild(c);
    }
    this.svg.setAttribute('viewBox', '0 0 8 8');
    this.svg.setAttribute('class', 'arrows');
    this.el.appendChild(this.svg);
    parent.appendChild(this.el);
    this.setOrientation('w');

    const at = (e: PointerEvent) => {
      const r = this.el.getBoundingClientRect();
      const col = Math.min(7, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * 8)));
      const row = Math.min(7, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * 8)));
      return this.orientation === 'w' ? row * 8 + col : 63 - (row * 8 + col);
    };
    this.el.addEventListener('pointerdown', (e) => this.onDown(at(e)));
    this.el.addEventListener('pointerup', (e) => this.onUp(at(e)));
  }

  setOrientation(o: 'w' | 'b') {
    this.orientation = o;
    this.cells.forEach((c, i) => {
      const d = o === 'w' ? i : 63 - i;
      c.style.gridRow = String(Math.floor(d / 8) + 1);
      c.style.gridColumn = String((d % 8) + 1);
      const col = d % 8, row = Math.floor(d / 8);
      (c.querySelector('.coord-r') as HTMLElement).textContent = col === 0 ? String(8 - (i >> 3)) : '';
      (c.querySelector('.coord-f') as HTMLElement).textContent = row === 7 ? 'abcdefgh'[i & 7] : '';
    });
  }

  get flipped() {
    return this.orientation === 'b';
  }

  setPieces(p: (string | null)[]) {
    p.forEach((v, i) => {
      if (this.shown[i] === v) return; // repainting all 32 outlined glyphs per move is what makes a board feel heavy
      this.shown[i] = v;
      const el = this.cells[i].querySelector('.piece') as HTMLElement;
      el.textContent = v ? pieceGlyph(v[1]) : '';
      el.className = 'piece' + (v ? ' ' + (v[0] === 'w' ? 'wp' : 'bp') : '');
    });
  }

  setHighlights(h: Highlights) {
    const targets = new Set(h.targets ?? []);
    this.cells.forEach((c, i) => {
      c.classList.toggle('selected', h.selected === i);
      c.classList.toggle('last', !!h.last && (h.last[0] === i || h.last[1] === i));
      c.classList.toggle('check', h.check === i);
      c.classList.toggle('target', targets.has(i));
      c.classList.toggle('capture', targets.has(i) && !!c.querySelector('.piece')?.textContent);
    });
  }

  /** Paint a heat-map under the pieces; values are 64 numbers or null to clear. */
  setHeat(values: ArrayLike<number> | null, color: ColorFn) {
    this.cells.forEach((c, i) => {
      const h = c.querySelector('.heat') as HTMLElement;
      h.style.background = values ? color(values[i]) : '';
    });
    this.el.classList.toggle('heated', !!values);
  }

  /** Small text in the corner of each square (null clears). */
  setLabels(labels: (string | null)[] | null) {
    this.cells.forEach((c, i) => {
      (c.querySelector('.lbl') as HTMLElement).textContent = labels?.[i] ?? '';
    });
  }

  private center(idx: number): [number, number] {
    const d = this.orientation === 'w' ? idx : 63 - idx;
    return [(d % 8) + 0.5, Math.floor(d / 8) + 0.5];
  }

  setArrows(arrows: Arrow[]) {
    this.svg.innerHTML = '';
    for (const a of arrows) {
      const [x1, y1] = this.center(a.from);
      const [x2, y2] = this.center(a.to);
      const dx = x2 - x1, dy = y2 - y1;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const w = a.width ?? 0.16;
      const head = w * 2.4;
      const bx = x2 - ux * head * 0.9, by = y2 - uy * head * 0.9;
      const g = document.createElementNS(SVG, 'g');
      g.setAttribute('opacity', String(a.opacity ?? 0.85));
      g.innerHTML =
        `<line x1="${x1 + ux * 0.2}" y1="${y1 + uy * 0.2}" x2="${bx}" y2="${by}" stroke="${a.color}" stroke-width="${w}" stroke-linecap="round"/>` +
        `<polygon fill="${a.color}" points="${x2 - ux * 0.08},${y2 - uy * 0.08} ${bx - uy * head * 0.75},${by + ux * head * 0.75} ${bx + uy * head * 0.75},${by - ux * head * 0.75}"/>`;
      this.svg.appendChild(g);
    }
  }
}

// ---- colour maps -----------------------------------------------------------------------------

const lerp = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t);
const STOPS = [
  [14, 12, 40], [58, 28, 120], [140, 40, 130], [225, 80, 70], [250, 190, 60], [253, 245, 170],
];
/** 0..1 -> dark purple through orange to pale yellow (activations are >= 0 after ReLU). */
export function heatColor(t: number, alpha = 0.92): string {
  t = Math.max(0, Math.min(1, t));
  const f = t * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(f));
  const [r, g, b] = lerp(STOPS[i], STOPS[i + 1], f - i);
  return `rgba(${r | 0},${g | 0},${b | 0},${alpha})`;
}

/** -1..1 -> red (negative) through transparent to green (positive). */
export function divergingColor(t: number): string {
  t = Math.max(-1, Math.min(1, t));
  const a = Math.abs(t) ** 0.7 * 0.85;
  return t >= 0 ? `rgba(52,211,153,${a})` : `rgba(248,86,86,${a})`;
}
