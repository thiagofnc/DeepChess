import './styles.css';
import { Net } from './engine/net';
import { Stockfish } from './engine/stockfish';
import { createPlay } from './play';
import { createExplore } from './explore';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

async function main() {
  let net: Net;
  try {
    net = await Net.load((s) => ($('loadTxt').textContent = s));
  } catch (e) {
    $('loadTxt').textContent = 'Could not start the neural network: ' + (e as Error).message;
    throw e;
  }
  $('backend').textContent = net.backend === 'webgpu' ? '⚡ WebGPU' : '🐢 WebAssembly (no WebGPU available)';
  $('loading').remove();

  const sf = new Stockfish();
  let explore: ReturnType<typeof createExplore> | null = null;
  let tab = 'play';
  const play = createPlay($('play'), { net, sf, onChange: () => tab === 'explore' && explore?.sync() });
  explore = createExplore($('explore'), net, play);

  document.querySelectorAll<HTMLElement>('#tabs button').forEach(
    (b) =>
      (b.onclick = () => {
        tab = b.dataset.tab!;
        document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('on', x === b));
        $('play').hidden = tab !== 'play';
        $('explore').hidden = tab !== 'explore';
        if (tab === 'explore') explore!.sync();
      }),
  );
}

main();
