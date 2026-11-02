import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const tests = [];

function test(name, run) {
  tests.push({ name, run });
}

try {
  const {
    BrowserSessionPersistence,
    BrowserSessionStateStore,
    SessionPersistenceError,
    validatePersistedHostedSession,
  } = await server.ssrLoadModule("/../drowse/webui/src/lib/sessionPersistence.ts");
  const legacyConversationsModule = await server.ssrLoadModule("/src/lib/runtime/legacyConversations.ts");
  const { LegacyStashingSessionStateStore, findLegacyAutosaves, clearLegacyAutosaves } =
    await server.ssrLoadModule("/src/lib/runtime/legacyAutosaves.ts");

  test("round-trips one authoritative session per model variant", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 1_000,
      runExclusive: serialRunner(),
    });
    const fastest = sessionInput("fastest", "session-a", 2);
    const balanced = sessionInput("balanced", "session-b", 4);

    const saved = await persistence.save(fastest, "owner-a");
    await persistence.save(balanced, "owner-a");
    saved.tree.rev = 99;
    saved.settings.temperature = 99;

    const restored = await persistence.claim(binding("fastest", "owner-a"));
    assert.equal(restored.status, "compatible");
    assert.equal(restored.record.tree.rev, 2);
    assert.equal(restored.record.settings.temperature, 0.7);
    const balancedRestored = await persistence.claim(binding("balanced", "owner-a"));
    assert.equal(balancedRestored.status, "compatible");
    assert.equal(balancedRestored.record.metadata.sessionId, "session-b");
    assert.equal(await persistence.delete("fastest"), true);
    assert.equal(await persistence.delete("fastest"), false);
    assert.equal((await persistence.claim(binding("fastest", "owner-a"))).status, "missing");
    assert.equal((await persistence.claim(binding("balanced", "owner-a"))).status, "compatible");
    await persistence.clear();
    assert.equal((await persistence.claim(binding("balanced", "owner-a"))).status, "missing");
  });

  test("round-trips browser position receipts and rejects malformed spans", async () => {
    const input = sessionInput("fastest", "span-session", 2);
    input.tree.nodes[1].recipe = {
      steering: "0.5 calm", sampling: null, thinking: false, seed: 1,
      probes: [], probe_hashes: {}, position_spans: [[1, 4], [10, 12]],
    };
    const persistence = new BrowserSessionPersistence({ store: memoryStore(), runExclusive: serialRunner() });
    await persistence.save(input, "owner-a");
    input.tree.nodes[1].recipe.position_spans[0][1] = 3;
    const restored = await persistence.claim(binding("fastest", "owner-a"));
    assert.deepEqual(restored.record.tree.nodes[1].recipe.position_spans, [[1, 4], [10, 12]]);
    for (const spans of [null, [[2, 1]], [[1, 4], [2, 3]], [[0, 1.5]]]) {
      input.tree.rev += 1;
      input.tree.nodes[1].recipe.position_spans = spans;
      await assert.rejects(persistence.save(input, "owner-a"), /position|Steering/);
    }
  });

  test("serializes writes and rejects stale revisions and relinquished owners", async () => {
    const store = memoryStore();
    const runner = trackedSerialRunner();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 2_000,
      runExclusive: runner.run,
    });
    await persistence.save(sessionInput("fastest", "session-a", 5), "owner-a");

    const newer = persistence.save(sessionInput("fastest", "session-a", 6), "owner-a");
    const stale = persistence.save(sessionInput("fastest", "session-a", 4), "owner-a");
    await newer;
    await assert.rejects(
      stale,
      (error) => error instanceof SessionPersistenceError &&
        error.code === "STALE_TREE_REVISION",
    );

    assert.equal(runner.maximumActive(), 1);
    const claimed = await persistence.claim(binding("fastest", "owner-b"));
    assert.equal(claimed.status, "compatible");
    assert.equal(claimed.record.tree.rev, 6);

    await assert.rejects(
      persistence.save(sessionInput("fastest", "session-a", 7), "owner-a"),
      (error) => error instanceof SessionPersistenceError &&
        error.code === "STALE_SESSION_OWNER",
    );
    await persistence.save(sessionInput("fastest", "session-a", 7), "owner-b");
    assert.equal(
      (await persistence.claim(binding("fastest", "owner-b"))).record.tree.rev,
      7,
    );
  });

  test("uses the named Web Lock for default cross-tab serialization", async () => {
    const requests = [];
    const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        locks: {
          async request(name, options, operation) {
            requests.push({ name, options });
            return operation();
          },
        },
      },
    });
    try {
      const persistence = new BrowserSessionPersistence({
        store: memoryStore(),
        now: () => 2_500,
      });
      await persistence.save(sessionInput("fastest", "session-a", 1), "owner-a");
      await persistence.claim(binding("fastest", "owner-a"));
      assert.deepEqual(requests, [
        { name: "drowse-hosted-session-state-v1", options: { mode: "exclusive" } },
        { name: "drowse-hosted-session-state-v1", options: { mode: "exclusive" } },
      ]);
    } finally {
      if (previous) Object.defineProperty(globalThis, "navigator", previous);
      else delete globalThis.navigator;
    }
  });

  test("preserves unsupported rows but removes corrupt current-schema rows", async () => {
    const unsupported = { schemaVersion: 99, future: true };
    const store = memoryStore(new Map([["fastest", unsupported]]));
    const recovered = [];
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
      onCorrupt: (modelVariantId, error) => recovered.push({ modelVariantId, error }),
    });

    assert.deepEqual(await persistence.claim(binding("fastest", "owner-a")), {
      status: "unsupported",
      schemaVersion: 99,
    });
    assert.deepEqual(store.rows.get("fastest"), unsupported);
    assert.equal(recovered.length, 0);

    store.rows.set("fastest", { schemaVersion: 2 });
    assert.equal((await persistence.claim(binding("fastest", "owner-a"))).status, "missing");
    assert.equal(store.rows.has("fastest"), false);
    assert.equal(recovered.length, 1);
    assert.equal(recovered[0].modelVariantId, "fastest");
    assert.match(recovered[0].error.message, /fields are invalid/);

    await persistence.save(sessionInput("fastest", "session-a", 1), "owner-a");
    assert.equal(
      (await persistence.claim(binding("fastest", "owner-a"))).record.tree.rev,
      1,
    );
  });

  test("explicit new-chat reset clears compatible autosaves without touching other models", async () => {
    for (const otherIdentity of [false, true]) {
      const store = memoryStore();
      const persistence = new BrowserSessionPersistence({ store, runExclusive: serialRunner() });
      await persistence.save(sessionInput("fastest", "session-a", 3), "owner-a");
      await persistence.save(sessionInput("balanced", "session-b", 4), "owner-a");
      const target = binding("fastest", "owner-b");
      if (otherIdentity) target.runtimeIdentitySha256 = "b".repeat(64);
      const result = await persistence.claim(target, "reset");
      assert.equal(result.status, "missing", "reset must not restore a compatible session");
      assert.equal(store.rows.has("fastest"), false);
      assert.equal(store.rows.get("balanced").tree.rev, 4);
    }
  });

  test("keeps incompatible runtime data until an explicit reset", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    await persistence.save(sessionInput("fastest", "session-a", 3), "owner-a");

    const runtimeMismatch = await persistence.claim({
      ...binding("fastest", "owner-b"),
      runtimeIdentitySha256: "b".repeat(64),
    });
    assert.equal(runtimeMismatch.status, "incompatible");
    assert.equal(runtimeMismatch.reason, "runtime_identity");
    assert.equal(store.rows.get("fastest").tree.rev, 3);

    const reset = await persistence.claim({
      ...binding("fastest", "owner-b"),
      runtimeIdentitySha256: "b".repeat(64),
    }, "reset");
    assert.equal(reset.status, "missing");
    assert.equal(store.rows.has("fastest"), false);
  });

  test("rejects malformed topology, non-JSON settings, and metadata mismatches", () => {
    const valid = persistedRecord("fastest", "session-a", 1);
    validatePersistedHostedSession(valid);

    const disconnected = structuredClone(valid);
    disconnected.tree.children_of.root = [];
    assert.throws(
      () => validatePersistedHostedSession(disconnected),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /absent from its parent's children row/.test(error.message),
    );

    const tensorSetting = structuredClone(valid);
    tensorSetting.settings.temperature = new Float32Array([1, 2]);
    assert.throws(
      () => validatePersistedHostedSession(tensorSetting),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /temperature is invalid/.test(error.message),
    );

    const mismatched = structuredClone(valid);
    mismatched.tree.model_id = "different/model";
    assert.throws(
      () => validatePersistedHostedSession(mismatched),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /does not match session metadata/.test(error.message),
    );

    const wrongFormat = structuredClone(valid);
    wrongFormat.tree.tree_format = 3;
    assert.throws(
      () => validatePersistedHostedSession(wrongFormat),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /format is unsupported/.test(error.message),
    );

    const wrongRoot = structuredClone(valid);
    wrongRoot.tree.nodes[0].role = "user";
    assert.throws(
      () => validatePersistedHostedSession(wrongRoot),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /system role/.test(error.message),
    );
  });

  test("round-trips vocabulary-sized alternatives without the old per-token cap", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({ store, runExclusive: serialRunner() });
    const input = sessionInput("fastest", "session-a", 1);
    const count = 262144;
    input.tree.nodes[1].recipe = {
      steering: null, thinking: null, seed: null, probes: [], probe_hashes: {},
      sampling: {
        temperature: 3, top_p: 1, top_k: count, max_tokens: 32768, seed: null,
        stop: null, logit_bias: null, presence_penalty: 0, frequency_penalty: 0,
        logprobs: null, return_hidden: false, return_top_k: count,
        user_role: null, assistant_role: null, persist_per_layer_scores: false,
        persist_subspace_coords: false, return_probe_readings: true,
      },
    };
    input.tree.nodes[1].tokens = [{
      token_id: 7, text: "Hello", logprob: -Math.log(count),
      top_alts: Array.from({ length: count }, (_, id) => ({ id, text: "token", logprob: -Math.log(count) })),
    }];
    await persistence.save(input, "owner-a");
    const restored = await persistence.claim(binding("fastest", "owner-a"));
    assert.equal(restored.status, "compatible");
    assert.equal(restored.record.tree.nodes[1].recipe.sampling.return_top_k, count);
    assert.equal(restored.record.tree.nodes[1].tokens[0].top_alts.length, count);
  });

  test("reads the 6.0 slim token row and refuses unknown or retired row fields", () => {
    const valid = persistedRecord("fastest", "session-a", 1);
    valid.tree.nodes[1].recipe = {
      steering: null, sampling: null, thinking: null, seed: null, probes: [], probe_hashes: {},
    };
    valid.tree.nodes[1].tokens = [{
      token_id: 7,
      text: "Hello",
      logprob: -0.25,
      perplexity: 1.28403,
      raw_index: 0,
      measurements: {
        version: 2,
        scope: "token",
        provenance: "captured",
        instruments: { geometry: { readings: { "local/calm.alert": { fraction: 0.5, coords: [0.25] } } } },
        scores: { "local/calm.alert": 0.25 },
      },
    }];
    valid.tree.nodes[1].aggregate_measurements = {
      version: 2,
      scope: "aggregate",
      provenance: "captured",
      instruments: {},
    };
    validatePersistedHostedSession(valid);

    for (const [field, value, pattern] of [
      ["producer_extension", { finite: 1 }, /unknown field producer_extension/],
      ["probes", { "local/calm.alert": 0.25 }, /earlier drowse/],
      ["per_layer_scores", { "4": { "local/calm.alert": 0.25 } }, /earlier drowse/],
    ]) {
      const invalid = structuredClone(valid);
      invalid.tree.nodes[1].tokens[0][field] = value;
      assert.throws(
        () => validatePersistedHostedSession(invalid),
        (error) => error.code === "INVALID_SESSION_STATE" && pattern.test(error.message),
        field,
      );
    }
    const fiveX = structuredClone(valid);
    fiveX.tree.nodes[1].tokens[0].measurements.version = 1;
    assert.throws(
      () => validatePersistedHostedSession(fiveX),
      (error) => error.code === "INVALID_SESSION_STATE" && /version 1/.test(error.message),
    );
    const noCapture = structuredClone(valid);
    delete noCapture.tree.nodes[1].prompt_capture;
    assert.throws(
      () => validatePersistedHostedSession(noCapture),
      (error) => error.code === "INVALID_SESSION_STATE",
    );
  });

  test("offers pre-v8 saved chats as one file without opening them", async () => {
    const rows = [
      { key: "old-b", value: { schemaVersion: 1, id: "old-b", name: "Later", modelId: "m", updatedAt: 20, snapshot: { version: 7, tree: { tree_format: 2 } } } },
      { key: "current", value: { schemaVersion: 1, id: "current", name: "Now", modelId: "m", updatedAt: 30, snapshot: { version: 8 } } },
      { key: "old-a", value: { schemaVersion: 1, id: "old-a", name: "Earlier", modelId: "m", updatedAt: 10, snapshot: { version: 6, extra: [1, 2] } } },
      { key: "broken", value: { schemaVersion: 1, id: "broken", snapshot: null } },
    ];
    let initialized = 0;
    const store = {
      async initialize() { initialized += 1; },
      async list() { return structuredClone(rows); },
      async read() { throw new Error("the export never opens a chat"); },
      async write() { throw new Error("the export never writes"); },
      async delete() { throw new Error("the export never deletes"); },
    };
    assert.equal(legacyConversationsModule.isLegacyConversationRow(rows[0].value), true);
    assert.equal(legacyConversationsModule.isLegacyConversationRow(rows[1].value), false);
    assert.equal(legacyConversationsModule.isLegacyConversationRow(rows[3].value), false);
    const autosave = legacyAutosave("fastest");
    const stash = memoryStash([{ key: "fastest@500", modelVariantId: "fastest", stashedAt: 500, record: autosave }]);
    const file = await legacyConversationsModule.exportLegacyConversations({
      library: store,
      autosaves: stash,
      now: Date.UTC(2026, 10, 1),
    });
    assert.equal(initialized, 1);
    assert.equal(file.filename, "drowse-chats-before-6.0-2026-11-01.json");
    assert.equal(file.mimeType, "application/json");
    assert.deepEqual(file.conversations.map((row) => [row.id, row.name, row.snapshotVersion]), [
      ["old-a", "Earlier", 6],
      ["old-b", "Later", 7],
    ]);
    const document = JSON.parse(file.json);
    assert.equal(document.format, "drowse-legacy-conversations");
    assert.equal(document.format_version, 1);
    assert.equal(document.exported_at, "2026-11-01T00:00:00.000Z");
    assert.deepEqual(document.conversations, [
      { id: "old-a", record: rows[2].value },
      { id: "old-b", record: rows[0].value },
    ]);
    assert.deepEqual(document.autosaves, [
      { model_variant_id: "fastest", stashed_at: "1970-01-01T00:00:00.500Z", record: autosave },
    ]);
    assert.deepEqual(file.autosaves, [{ modelVariantId: "fastest", stashedAt: 500, updatedAt: 1_000 }]);
    const empty = { ...store, async list() { return [rows[1]]; } };
    assert.equal(
      await legacyConversationsModule.exportLegacyConversations({ library: empty, autosaves: memoryStash() }),
      null,
    );
    const onlyAutosaves = await legacyConversationsModule.exportLegacyConversations({ library: empty, autosaves: stash });
    assert.deepEqual(JSON.parse(onlyAutosaves.json).conversations, []);
    assert.equal(onlyAutosaves.autosaves.length, 1);
  });

  test("stashes 5.x autosaves instead of deleting them", async () => {
    const legacy = legacyAutosave("fastest");
    const inner = memoryStore(new Map([["fastest", legacy], ["balanced", legacyAutosave("balanced")]]));
    const stash = memoryStash();
    let clock = 700;
    const recovered = [];
    const persistence = new BrowserSessionPersistence({
      store: new LegacyStashingSessionStateStore(inner, stash, () => clock++),
      runExclusive: serialRunner(),
      onCorrupt: (modelVariantId) => recovered.push(modelVariantId),
    });

    // claim: the 5.x row fails 6.0 validation, so it leaves the session store but survives in the stash.
    assert.equal((await persistence.claim(binding("fastest", "owner-a"))).status, "missing");
    assert.equal(inner.rows.has("fastest"), false);
    assert.deepEqual(recovered, ["fastest"]);
    assert.deepEqual(await findLegacyAutosaves(stash), [
      { key: "fastest@700", modelVariantId: "fastest", stashedAt: 700, record: legacy },
    ]);

    // reset of a 5.x row stashes it too; a 6.0 row deleted or reset is not stashed.
    assert.equal((await persistence.claim(binding("balanced", "owner-a"), "reset")).status, "missing");
    await persistence.save(sessionInput("fastest", "session-a", 1), "owner-a");
    assert.equal((await persistence.claim(binding("fastest", "owner-b"), "reset")).status, "missing");
    await persistence.save(sessionInput("fastest", "session-a", 1), "owner-b");
    assert.equal(await persistence.delete("fastest"), true);
    assert.deepEqual((await findLegacyAutosaves(stash)).map((entry) => entry.key), ["fastest@700", "balanced@701"]);

    // save over a 5.x row stashes it before replacing it.
    inner.rows.set("other", legacyAutosave("other"));
    await persistence.save(sessionInput("other", "session-c", 1), "owner-a");
    assert.equal(inner.rows.get("other").tree.rev, 1);
    assert.deepEqual((await findLegacyAutosaves(stash)).map((entry) => entry.modelVariantId), ["fastest", "balanced", "other"]);

    // a failed copy keeps the record and fails the claim.
    inner.rows.set("kept", legacyAutosave("kept"));
    const failing = new BrowserSessionPersistence({
      store: new LegacyStashingSessionStateStore(inner, { ...memoryStash(), async put() { throw new Error("quota"); } }),
      runExclusive: serialRunner(),
    });
    await assert.rejects(() => failing.claim(binding("kept", "owner-a")), /quota/);
    assert.equal(inner.rows.has("kept"), true);

    await clearLegacyAutosaves(stash);
    assert.deepEqual(await findLegacyAutosaves(stash), []);
    stash.entries.set("x@1", { key: "x@1", modelVariantId: "x", stashedAt: 1, record: {} });
    await persistence.clear();
    assert.equal(inner.rows.size, 0);
    assert.equal(stash.entries.size, 0);
  });

  test("reopens the IndexedDB state store after versionchange", async () => {
    const fake = fakeIndexedDb();
    const previous = Object.getOwnPropertyDescriptor(globalThis, "indexedDB");
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: fake,
    });
    try {
      const store = new BrowserSessionStateStore();
      const record = persistedRecord("fastest", "session-a", 1);
      await Promise.all([store.initialize(), store.initialize()]);
      assert.equal(fake.openCalls, 1);
      await store.write(record);
      assert.equal((await store.read("fastest")).tree.rev, 1);

      fake.handles[0].onversionchange();
      assert.equal(fake.handles[0].closed, true);
      assert.equal((await store.read("fastest")).tree.rev, 1);
      assert.equal(fake.openCalls, 2);
      assert.equal(fake.handles[1].closed, false);
    } finally {
      if (previous) Object.defineProperty(globalThis, "indexedDB", previous);
      else delete globalThis.indexedDB;
    }
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} session persistence tests passed`);
} finally {
  await server.close();
}

function sessionInput(modelVariantId, sessionId, revision) {
  return {
    modelVariantId,
    metadata: {
      sessionId,
      modelId: "HuggingFaceTB/SmolLM2-360M-Instruct",
      runtimeIdentitySha256: "a".repeat(64),
      contextTokens: 2_048,
      createdAt: 100,
    },
    settings: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      max_tokens: 256,
      system_prompt: "You are helpful.",
      thinking: false,
    },
    tree: tree(sessionId, revision),
  };
}

function persistedRecord(modelVariantId, sessionId, revision) {
  return {
    ...sessionInput(modelVariantId, sessionId, revision),
    schemaVersion: 2,
    ownerEpoch: "owner-a",
    updatedAt: 1_000,
  };
}

function binding(modelVariantId, ownerEpoch) {
  return {
    modelVariantId,
    runtimeIdentitySha256: "a".repeat(64),
    contextTokens: 2_048,
    ownerEpoch,
  };
}

function tree(sessionId, revision) {
  const root = node("root", null, "system", "");
  const user = node("user-1", "root", "user", "Hello");
  return {
    tree_format: 2,
    drowse_version: "fixture",
    model_id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    session_id: sessionId,
    name: "Local conversation",
    rev: revision,
    root_id: root.id,
    active_node_id: user.id,
    nodes: [root, user],
    children_of: { root: [user.id], [user.id]: [] },
    cast: {},
  };
}

function node(id, parentId, role, text) {
  return {
    id,
    parent_id: parentId,
    role,
    text,
    role_label: null,
    thinking_text: null,
    aggregate_readings: {},
    aggregate_measurements: null,
    applied_steering: null,
    finish_reason: null,
    starred: false,
    notes: "",
    created_at: 100,
    edited_at: null,
    edit_count: 0,
    mean_logprob: null,
    mean_surprise: null,
    recipe: null,
    tokens: null,
    thinking_tokens: null,
    raw_token_ids: null,
    prompt_capture: null,
  };
}

function memoryStore(initial = new Map()) {
  const rows = initial;
  return {
    rows,
    async initialize() {},
    async read(key) {
      const value = rows.get(key);
      return value === undefined ? undefined : structuredClone(value);
    },
    async write(record) {
      rows.set(record.modelVariantId, structuredClone(record));
    },
    async delete(key) {
      return rows.delete(key);
    },
    async clear() {
      rows.clear();
    },
  };
}

// A 5.x hosted autosave: schemaVersion 2 like 6.0, with 5.x token rows and no prompt_capture.
function legacyAutosave(modelVariantId) {
  const record = persistedRecord(modelVariantId, "session-5x", 7);
  for (const entry of record.tree.nodes) delete entry.prompt_capture;
  record.tree.nodes[1].tokens = [{ token_id: 1, text: "Hello", probes: { a: 0.5 }, per_layer_scores: {} }];
  return record;
}

function memoryStash(initial = []) {
  const entries = new Map(initial.map((entry) => [entry.key, structuredClone(entry)]));
  return {
    entries,
    async initialize() {},
    async put(entry) {
      entries.set(entry.key, structuredClone(entry));
    },
    async list() {
      return [...entries.values()].map((entry) => structuredClone(entry));
    },
    async clear() {
      entries.clear();
    },
  };
}

function serialRunner() {
  let tail = Promise.resolve();
  return async (operation) => {
    const predecessor = tail;
    let release;
    tail = new Promise((resolve) => {
      release = resolve;
    });
    await predecessor;
    try {
      return await operation();
    } finally {
      release();
    }
  };
}

function trackedSerialRunner() {
  let active = 0;
  let maximum = 0;
  const serial = serialRunner();
  return {
    run: (operation) => serial(async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await Promise.resolve();
      try {
        return await operation();
      } finally {
        active -= 1;
      }
    }),
    maximumActive: () => maximum,
  };
}

function fakeIndexedDb() {
  const rows = new Map();
  let storeCreated = false;
  return {
    openCalls: 0,
    handles: [],
    open() {
      this.openCalls += 1;
      const handle = fakeDatabaseHandle(rows, () => storeCreated, () => {
        storeCreated = true;
      });
      this.handles.push(handle);
      const request = { result: handle, error: null };
      queueMicrotask(() => {
        if (!storeCreated) request.onupgradeneeded?.();
        request.onsuccess?.();
      });
      return request;
    },
  };
}

function fakeDatabaseHandle(rows, hasStore, createStore) {
  const handle = {
    closed: false,
    onversionchange: null,
    onclose: null,
    objectStoreNames: { contains: () => hasStore() },
    createObjectStore() {
      createStore();
    },
    close() {
      this.closed = true;
    },
    transaction() {
      if (this.closed) throw new DOMException("Database is closed", "InvalidStateError");
      const transaction = {
        error: null,
        objectStore() {
          return {
            get(key) {
              const request = { result: undefined, error: null };
              queueMicrotask(() => {
                const value = rows.get(key);
                request.result = value === undefined ? undefined : structuredClone(value);
                request.onsuccess?.();
                queueMicrotask(() => transaction.oncomplete?.());
              });
              return request;
            },
            put(record) {
              rows.set(record.modelVariantId, structuredClone(record));
              queueMicrotask(() => transaction.oncomplete?.());
            },
            delete(key) {
              rows.delete(key);
              queueMicrotask(() => transaction.oncomplete?.());
            },
            clear() {
              rows.clear();
              queueMicrotask(() => transaction.oncomplete?.());
            },
          };
        },
      };
      return transaction;
    },
  };
  return handle;
}
