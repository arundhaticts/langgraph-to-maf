import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' so the built assets load when FastAPI serves dist/ at '/'.
// dev proxy forwards /api to the FastAPI backend on port 8000.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
