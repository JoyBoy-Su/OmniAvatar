import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8080,
    host: '0.0.0.0',
    proxy: {
      '/socket.io': {
        target: 'http://localhost:20143',
        changeOrigin: true,
        ws: true,
      },
      '/api': {
        target: 'http://localhost:20143',
        changeOrigin: true,
      },
    },
  },
})

