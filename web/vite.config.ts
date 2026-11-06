import { defineConfig } from 'vite';

// base './' lets the built site run from any sub-path (/, /deepchess/, a GitHub Pages repo path, ...).
export default defineConfig({
  base: './',
  // Use ONNX Runtime's "extern wasm" entry: the .wasm is served from public/vendor/ort instead of being bundled twice.
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  worker: { format: 'es' },
});
