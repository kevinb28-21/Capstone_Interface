import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5182,
    host: true, // Allow external connections
    proxy: {
      '/api': 'http://localhost:5050',
      '/uploads': 'http://localhost:5050'
    }
  },
  // Production build settings
  build: {
    outDir: 'dist',
    sourcemap: false
  }
});


