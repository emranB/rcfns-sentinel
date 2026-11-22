import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/status': 'http://127.0.0.1:9388',
      '/chat': 'http://127.0.0.1:9388',
      '/sources': 'http://127.0.0.1:9388',
    },
  },
});
