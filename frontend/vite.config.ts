import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 6000,
    rollupOptions: {
      output: {
        // Plotly를 별도 청크로 분리 → 초기 앱 청크 경량화, 캐시 분리
        manualChunks(id: string) {
          if (id.includes("node_modules/plotly.js") || id.includes("node_modules/react-plotly.js")) {
            return "plotly";
          }
          return undefined;
        },
      },
    },
  },
});
