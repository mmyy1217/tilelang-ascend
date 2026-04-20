import { defineConfig } from "vitepress"

export default defineConfig({
  title: "MLIR Pipeline Graphs",
  description: "Generated documentation for bishengir MLIR pipelines.",
  srcDir: ".",
  themeConfig: {
    nav: [{ text: "Home", link: "/" }],
    search: {
      provider: "local",
    },
  },
})
