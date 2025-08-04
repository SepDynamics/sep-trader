import { defineConfig } from 'vite'

export default defineConfig({
  base: '/',
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: './index.html',
        investors: './investors.html'
      }
    }
  },
  server: {
    port: 3000,
    open: true
  },
  esbuild: {
    supported: {
      'top-level-await': true
    }
  }
})
