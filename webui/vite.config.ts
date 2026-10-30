import { defineConfig, mergeConfig } from "vite";
import dashboard from "../drowse/webui/vite.config";

// Ad-hoc Vite servers in scripts/ (createServer without a configFile) resolve
// this file, so shared modules load with the dashboard's own plugins and runtime
// seams. Builds go to an ignored local directory, never into the submodule.
// The dashboard config is a function of the command, so resolve it first.
export default defineConfig(async (env) => mergeConfig(await dashboard(env), {
  build: { outDir: "dist-local", emptyOutDir: true },
}));
