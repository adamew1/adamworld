import { defineConfig } from "vite";

// Dev server proxies API calls to the FastAPI backend on :8000 so the viewer
// can `fetch("/worlds/<id>")` without CORS fuss during development.
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/worlds": "http://127.0.0.1:8000",
      "/jobs": "http://127.0.0.1:8000",
    },
  },
  build: {
    target: "es2022",
    sourcemap: true,
    rollupOptions: {
      input: {
        main: "index.html",
        editor: "editor.html",
      },
    },
  },
});
