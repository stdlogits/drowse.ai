import { mergeConfig } from "vite";
import dashboard from "../drowse/webui/vite.config";

// Ad-hoc Vite servers in scripts/ (createServer without a configFile) resolve
// this file, so shared modules load with the dashboard's own plugins and runtime
// seams. Builds go to an ignored local directory, never into the submodule.
export default mergeConfig(dashboard, {
  build: { outDir: "dist-local", emptyOutDir: true },
});
