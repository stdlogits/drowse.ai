<script module lang="ts">
  /** Bump when `public-hosted/video/ethereal-orb*.mp4` is re-encoded; `/video/*` is cached as immutable. */
  export const HERO_VIDEO_VERSION = "2026-10-29";
  export const HERO_MOTION_STORAGE_KEY = "drowse.hero-motion";
  const READY_TIMEOUT_MS = 10_000;

  type NetworkHints = { saveData?: boolean; effectiveType?: string };

  /** Slow links and low-memory devices keep the cached poster instead of streaming video. */
  export function heroVideoAllowed(connection: NetworkHints | undefined, deviceMemory: number | undefined): boolean {
    if (/^(?:slow-2g|2g|3g)$/.test(connection?.effectiveType ?? "")) return false;
    return !(typeof deviceMemory === "number" && deviceMemory <= 2);
  }

  export function heroVideoUrl(width: number): string {
    return `${width <= 760 ? "/video/ethereal-orb-720.mp4" : "/video/ethereal-orb.mp4"}?v=${HERO_VIDEO_VERSION}`;
  }

  function readMotionPreference(): boolean {
    try { return localStorage.getItem(HERO_MOTION_STORAGE_KEY) === "paused"; } catch { return false; }
  }

  function writeMotionPreference(paused: boolean) {
    try {
      if (paused) localStorage.setItem(HERO_MOTION_STORAGE_KEY, "paused");
      else localStorage.removeItem(HERO_MOTION_STORAGE_KEY);
    } catch { /* The choice still applies to this page without storage. */ }
  }
</script>

<script lang="ts">
  import { onMount } from "svelte";
  import { IconButton } from "@stdlogits/ledger";
  import { theme } from "../../../../drowse/webui/src/lib/theme";
  import type { HeroPalette } from "./heroShaderRuntime";

  // The landing's orb, contained in its frame (the hero beside the copy). The scene is rendered
  // from recorded light, then recoloured onto Ledger's paper and depth ramp (--deep to --late) for
  // the current theme; the still artwork does the same with the ramp masked by the orb's light.
  // Whether a moving scene belongs in Ledger's flat look is a9's call.
  let { paused = false, motionControl = false }: {
    paused?: boolean;
    /** Show a pause/play control. Only enable where the control is not inside an aria-hidden ancestor. */
    motionControl?: boolean;
  } = $props();
  let updateMotion = $state<(() => void) | null>(null);
  let redraw: (() => void) | null = null;
  let userPaused = $state(false);
  let controlAvailable = $state(false);
  $effect(() => { void paused; void userPaused; updateMotion?.(); });
  // A theme change recolours the next frame (and the still one at once).
  $effect(() => { void theme.mode; redraw?.(); });
  let host: HTMLElement;
  let canvas: HTMLCanvasElement;
  let status = $state<"fallback" | "loading" | "ready">("loading");

  /** One token as a 0..1 RGB triple, read through a canvas so any CSS colour form parses. */
  function tokenRgb(name: string, probe: CanvasRenderingContext2D | null): number[] {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    if (!probe || !value) return [0, 0, 0];
    probe.fillStyle = "transparent";
    probe.fillStyle = value;
    const color = String(probe.fillStyle);
    const hex = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(color);
    if (hex) return hex.slice(1, 4).map((part) => parseInt(part, 16) / 255);
    const channels = /(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)/.exec(color);
    return channels ? channels.slice(1, 4).map((part) => Number(part) / 255) : [0, 0, 0];
  }

  function readPalette(): HeroPalette {
    const probe = document.createElement("canvas").getContext("2d");
    return {
      paper: tokenRgb("--paper", probe), ink: tokenRgb("--ink", probe),
      deep: tokenRgb("--deep", probe), late: tokenRgb("--late", probe),
      dark: theme.mode === "dark",
    };
  }

  function toggleMotion() {
    userPaused = !userPaused;
    writeMotionPreference(userPaused);
  }

  onMount(() => {
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    const hints = navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string }; deviceMemory?: number };
    const saveData = hints.connection?.saveData;
    const videoAllowed = heroVideoAllowed(hints.connection, hints.deviceMemory);
    const video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "none";
    const poster = new Image();
    let source: ReturnType<typeof import("./heroShaderRuntime").createHeroShaderSource> | null = null;
    let disposed = false, initialized = false, inView = false, lost = false, failed = false, videoRequested = false;
    let frame = 0, lastFrame = 0, progress = 0, scrollFrame = 0;
    let readyTimer: ReturnType<typeof setTimeout> | undefined;
    let cancelVideoSchedule: (() => void) | null = null;
    let palette = readPalette();
    userPaused = readMotionPreference();
    const refreshControl = () => { controlAvailable = motionControl && !motion.matches && !saveData && !failed; };
    refreshControl();
    if (motion.matches || saveData || userPaused) status = "fallback";
    const moving = () => !disposed && !paused && !userPaused && !motion.matches && !saveData && !document.hidden && inView && !lost;
    const stop = () => {
      cancelAnimationFrame(frame);
      frame = 0; lastFrame = 0;
      video.pause();
    };
    const fail = () => {
      clearTimeout(readyTimer);
      cancelVideoSchedule?.();
      stop(); source?.dispose(); source = null; status = "fallback";
      failed = true; refreshControl();
    };
    const playVideo = () => {
      if (!videoRequested) return;
      void video.play().then(() => { if (!moving()) video.pause(); }).catch(() => undefined);
    };
    const render = (now: number) => {
      frame = 0;
      if (!moving() || !source) return;
      if (!lastFrame || now - lastFrame >= source.frameInterval) {
        const delta = lastFrame ? Math.min((now - lastFrame) / 1000, 0.1) : 1 / 30;
        lastFrame = now;
        try {
          if (source.update(progress, delta, palette)) { status = "ready"; clearTimeout(readyTimer); }
        } catch { fail(); return; }
      }
      frame = requestAnimationFrame(render);
    };
    const start = () => {
      if (!source || !moving() || frame) return;
      // Video adds moving light; the renderer also works with the cached artwork.
      playVideo();
      frame = requestAnimationFrame(render);
    };
    const onScroll = () => {
      // The camera travels into the orb as the page scrolls; the still artwork stays where it is.
      if (paused || userPaused || motion.matches || saveData || status === "fallback") return;
      progress = Math.min(1, Math.max(0, window.scrollY / Math.max(innerHeight * 1.8, 1)));
      host.dataset.travel = progress.toFixed(3);
    };
    const queueScroll = () => {
      if (scrollFrame) return;
      scrollFrame = requestAnimationFrame(() => { scrollFrame = 0; onScroll(); });
    };
    const resize = () => {
      onScroll();
      if (source) {
        try {
          source.resize(host.clientWidth, host.clientHeight);
          // Resizing clears the canvas; redraw before Safari can composite it.
          source.update(progress, 0, palette);
        } catch { fail(); }
      }
    };
    redraw = () => {
      palette = readPalette();
      try { source?.update(progress, 0, palette); }
      catch { fail(); }
    };
    const loadVideo = () => {
      cancelVideoSchedule = null;
      if (disposed || !source || videoRequested) return;
      videoRequested = true;
      video.src = heroVideoUrl(host.clientWidth);
      video.load();
      if (moving()) playVideo();
    };
    const scheduleVideo = () => {
      if (!videoAllowed || videoRequested || cancelVideoSchedule) return;
      // The poster renders first; video streams only after the page and its idle work settle.
      let idle = 0, timer: ReturnType<typeof setTimeout> | undefined;
      const whenIdle = () => {
        window.removeEventListener("load", whenIdle);
        if (typeof requestIdleCallback === "function") idle = requestIdleCallback(loadVideo, { timeout: 3000 });
        else timer = setTimeout(loadVideo, 1500);
      };
      cancelVideoSchedule = () => {
        window.removeEventListener("load", whenIdle);
        if (idle) cancelIdleCallback(idle);
        clearTimeout(timer);
        cancelVideoSchedule = null;
      };
      if (document.readyState === "complete") whenIdle();
      else window.addEventListener("load", whenIdle);
    };
    const initialize = async () => {
      if (initialized || failed || !moving()) return;
      initialized = true;
      status = "loading";
      try {
        const { createHeroShaderSource } = await import("./heroShaderRuntime");
        if (disposed || lost) return;
        if (!moving()) { initialized = false; status = "fallback"; return; }
        poster.onerror = fail;
        poster.src = "/images/ethereal-orb.jpg";
        source = createHeroShaderSource(canvas, video, poster);
        resize();
        if (!source) return;
        // Without a decodable poster or video frame the renderer can never draw.
        readyTimer = setTimeout(() => { if (status === "loading") fail(); }, READY_TIMEOUT_MS);
        scheduleVideo();
        start();
      } catch { fail(); }
    };
    updateMotion = () => {
      if (!moving()) {
        stop();
        if ((!source || status === "loading") && (motion.matches || saveData || userPaused)) status = "fallback";
      } else {
        onScroll();
        void initialize();
        start();
      }
    };
    const update = () => { refreshControl(); updateMotion?.(); };
    const contextLost = (event: Event) => {
      event.preventDefault(); lost = true; fail();
    };
    const contextRestored = () => {
      lost = false; failed = false; initialized = false; refreshControl(); void initialize();
    };
    const intersection = new IntersectionObserver(([entry]) => {
      inView = Boolean(entry?.isIntersecting); update();
    });
    const observer = new ResizeObserver(resize);
    intersection.observe(host); observer.observe(host);
    motion.addEventListener("change", update);
    document.addEventListener("visibilitychange", update);
    window.addEventListener("scroll", queueScroll, { passive: true });
    window.addEventListener("pagehide", stop);
    window.addEventListener("pageshow", update);
    canvas.addEventListener("webglcontextlost", contextLost);
    canvas.addEventListener("webglcontextrestored", contextRestored);
    return () => {
      disposed = true; stop(); source?.dispose(); source = null;
      clearTimeout(readyTimer);
      cancelVideoSchedule?.();
      cancelAnimationFrame(scrollFrame);
      redraw = null;
      intersection.disconnect(); observer.disconnect();
      motion.removeEventListener("change", update);
      document.removeEventListener("visibilitychange", update);
      window.removeEventListener("scroll", queueScroll);
      window.removeEventListener("pagehide", stop);
      window.removeEventListener("pageshow", update);
      canvas.removeEventListener("webglcontextlost", contextLost);
      canvas.removeEventListener("webglcontextrestored", contextRestored);
      if (videoRequested) { video.removeAttribute("src"); video.load(); }
      poster.onerror = null;
      poster.removeAttribute("src");
      updateMotion = null;
    };
  });
</script>

<div class="hero-frame">
  <div class="hero-visual contained" bind:this={host} data-shader-status={status} aria-hidden="true">
    <div class:visible={status === "fallback"} class="fallback"></div>
    <canvas bind:this={canvas} class:visible={status === "ready"}></canvas>
  </div>
  {#if controlAvailable}
    <div class="motion-toggle">
      <IconButton icon={userPaused ? "play" : "pause"} label={userPaused ? "Play background animation" : "Pause background animation"} onclick={toggleMotion} />
    </div>
  {/if}
</div>

<style>
  /* The orb fills its frame on the paper; the still artwork is the depth ramp seen through the
     orb's light. Both arrive with Ledger's fade. */
  .hero-frame { position: absolute; inset: 0; }
  .hero-visual { position: absolute; inset: 0; overflow: hidden; isolation: isolate; border-radius: var(--r-row); background: var(--paper); pointer-events: none; }
  .fallback, canvas { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; transition: opacity var(--in) var(--ease); }
  .fallback { background: linear-gradient(90deg, var(--deep), var(--late)); -webkit-mask: url("/images/ethereal-orb.jpg") center / cover no-repeat luminance; mask: url("/images/ethereal-orb.jpg") center / cover no-repeat luminance; }
  .fallback.visible, canvas.visible { opacity: 1; }
  canvas { display: block; }
  .motion-toggle { position: absolute; z-index: 2; right: var(--s3); bottom: var(--s3); }
</style>
