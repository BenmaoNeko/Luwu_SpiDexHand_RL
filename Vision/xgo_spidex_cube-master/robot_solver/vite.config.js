import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { defineConfig } from "vite";

const require = createRequire(import.meta.url);

// cube.js 1.3.2's CoffeeScript UMD solver reads this.Cube at module scope.
// Modern ESM has undefined top-level `this`. Wrap the unmodified algorithms
// with explicit UMD contexts at build time (no runtime eval, no solver fork).
function cubejsCompatibility() {
  const id = "\0cubejs-browser-compat";
  return {
    name: "cubejs-umd-context",
    enforce: "pre",
    resolveId(source) {
      if (source === "cubejs") return id;
    },
    load(source) {
      if (source !== id) return null;
      const root = dirname(require.resolve("cubejs"));
      const core = readFileSync(join(root, "lib/cube.js"), "utf8");
      const original = readFileSync(join(root, "lib/solve.js"), "utf8");
      const lookup = "Cube = this.Cube || require('./cube');";
      if (!original.includes(lookup))
        throw new Error(
          "cube.js adapter needs review for the installed version",
        );
      const solver = original.replace(lookup, "Cube = this.Cube;");
      return `const module = { exports: {} };\n(function(){\n${core}\n}).call(module.exports);\nconst Cube = module.exports;\n(function(){\n${solver}\n}).call({Cube});\nexport default Cube;`;
    },
  };
}

export default defineConfig({
  plugins: [cubejsCompatibility()],
  worker: { plugins: () => [cubejsCompatibility()] },
  optimizeDeps: { exclude: ["cubejs"] },
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [{ name: "three", test: /node_modules[\\/]three/ }],
        },
      },
    },
  },
});
