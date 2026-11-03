import { defineConfig, devices } from "@playwright/test";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const host = "127.0.0.1";
const devPort = 4176;
const previewPort = 4177;
const browserChannel = process.env.PLAYWRIGHT_CHANNEL;
const runCacheRoot = resolve(tmpdir(), `drowse-playwright-vite-${process.pid}`);
// PLAYWRIGHT_REUSE_SERVERS=1 attaches to dev and preview servers already listening on the fixed
// ports (fast spec-by-spec runs); without it every run starts and builds its own.
const reuseServers = process.env.PLAYWRIGHT_REUSE_SERVERS === "1";
// Chromium runs by default; E2E_WEBKIT=1 adds the iPhone layout pass on WebKit.
const webkit = process.env.E2E_WEBKIT === "1";

export default defineConfig({
  testDir: "./e2e",
  outputDir: resolve(tmpdir(), "drowse-playwright-results"),
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  timeout: 60_000,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.E2E_WORKERS ? Number(process.env.E2E_WORKERS) : 4,
  reporter: process.env.CI ? [["github"], ["line"]] : "line",
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://${host}:${previewPort}`,
    serviceWorkers: "allow",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(browserChannel ? { channel: browserChannel } : {}),
      },
    },
    ...(webkit ? [{
      name: "webkit-ios-layout",
      // "ios-layout" also matches every *.ios-layout.spec.ts.
      testMatch: /(ios-layout|hosted-onboarding|site-pages|chat-library|technical-controls|completion-selection|chat-name-stability|loom-search|token-inspector|mobile-sheet|model-downloads|new-chat-start|storage-persistence|theme-default|public-header|saved-chat-menu|contact|workbench-rooms|sae-descriptions)\.spec\.ts/,
      use: {
        ...devices["iPhone 13"],
      },
    }] : []),
  ],
  webServer: [
    {
      command: `npm run dev:hosted -- --force --host ${host} --port ${devPort} --strictPort`,
      url: `http://${host}:${devPort}`,
      env: { DROWSE_VITE_CACHE_DIR: resolve(runCacheRoot, "dev") },
      reuseExistingServer: reuseServers,
      timeout: 120_000,
    },
    {
      command: `npm run build:hosted && npm run preview:hosted -- --host ${host} --port ${previewPort} --strictPort`,
      url: `http://${host}:${previewPort}`,
      env: { DROWSE_VITE_CACHE_DIR: resolve(runCacheRoot, "preview") },
      reuseExistingServer: reuseServers,
      timeout: 120_000,
    },
  ],
});
