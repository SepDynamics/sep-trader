import { defineConfig } from 'vite'

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: './index.html',
        platform: './platform.html',
        investors: './investors.html',
        technology: './technology.html',
        demo: './demo.html',
        contact: './contact.html'
      }
    }
  },
  server: {
    port: 3000,
    open: true
  }
})
