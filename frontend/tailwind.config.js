/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "Consolas", "monospace"]
      },
      colors: {
        graphite: "#080b11",
        panel: "#101620",
        line: "#263142",
        samsung: "#3f7cff"
      }
    }
  },
  plugins: []
};

