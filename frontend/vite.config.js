import { defineConfig } from 'vite' // Vite configuration helper
import react from '@vitejs/plugin-react' // React plugin for Vite

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Proxy API requests to the FastAPI backend running on port 8000
    // This avoids CORS issues during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '') // Removes '/api' prefix before sending to backend
      }
    }
  }
})
