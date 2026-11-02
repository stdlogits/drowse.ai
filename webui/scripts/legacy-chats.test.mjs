// The old-chats notice in Your chats: chats saved before Drowse 6.0 (pre-v8 library rows and the
// 5.x autosaves the worker stashed) are counted, offered as one download and dismissable on this
// device without deleting anything. Runs the notice's module over in-memory stores and checks how
// HostedHome uses it.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createServer } from "vite";

const server = await createServer({ appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
const notice = await server.ssrLoadModule("/src/hosted/ui/legacyChats.ts");
test.after(() => server.close());

const rows = [
  { key: "old-b", value: { schemaVersion: 1, id: "old-b", name: "Later", modelId: "m", updatedAt: 20, snapshot: { version: 7 } } },
  { key: "current", value: { schemaVersion: 1, id: "current", name: "Now", modelId: "m", updatedAt: 30, snapshot: { version: 8 } } },
  { key: "old-a", value: { schemaVersion: 1, id: "old-a", name: "Earlier", modelId: "m", updatedAt: 10, snapshot: { version: 6 } } },
];

function library(list = rows) {
  const touched = [];
  return {
    touched,
    async initialize() {},
    async list() { return structuredClone(list); },
    async read() { touched.push("read"); },
    async write() { touched.push("write"); },
    async delete() { touched.push("delete"); return true; },
    close() { touched.push("close"); },
  };
}

function stash(entries = []) {
  const touched = [];
  return {
    touched,
    async initialize() {},
    async put() { touched.push("put"); },
    async list() { return structuredClone(entries); },
    async clear() { touched.push("clear"); },
    close() { touched.push("close"); },
  };
}

function storage() {
  const values = new Map();
  return { values, getItem: (key) => values.get(key) ?? null, setItem: (key, value) => values.set(key, String(value)) };
}

const autosave = { key: "fastest@500", modelVariantId: "fastest", stashedAt: 500, record: { variant: "fastest", updatedAt: 1_000 } };

test("counts old saved chats and stashed autosaves without opening, changing or closing given stores", async () => {
  const store = library();
  const stashed = stash([autosave]);
  const chats = await notice.findLegacyChats({ library: store, autosaves: stashed });
  assert.deepEqual(chats.conversationIds, ["old-a", "old-b"]);
  assert.deepEqual(chats.keys, ["chat:old-a", "chat:old-b", "autosave:fastest@500"]);
  assert.deepEqual(store.touched, [], "the count reads the list only");
  assert.deepEqual(stashed.touched, []);
  assert.deepEqual(await notice.findLegacyChats({ library: library([rows[1]]), autosaves: stash() }), { keys: [], conversationIds: [] });
});

test("says how many chats from before Drowse 6.0 can't be opened", () => {
  assert.equal(notice.legacyChatsTitle(1), "1 chat from before Drowse 6.0 can’t be opened here");
  assert.equal(notice.legacyChatsTitle(3), "3 chats from before Drowse 6.0 can’t be opened here");
});

test("dismissing hides the notice on this device, deletes nothing, and a newly stashed chat brings it back", async () => {
  const local = storage();
  const store = library();
  const stashed = stash([autosave]);
  const chats = await notice.findLegacyChats({ library: store, autosaves: stashed });
  assert.equal(notice.legacyNoticeShown(chats, notice.readDismissedLegacyChats(local)), true);
  const dismissed = notice.dismissLegacyChats(chats, local);
  assert.equal(notice.legacyNoticeShown(chats, dismissed), false);
  assert.equal(notice.legacyNoticeShown(chats, notice.readDismissedLegacyChats(local)), false, "the dismissal is stored");
  assert.match(notice.LEGACY_CHATS_DISMISSED_KEY, /^drowse\./, "clearing all local data clears the dismissal with the stash");
  assert.deepEqual([...local.values.keys()], [notice.LEGACY_CHATS_DISMISSED_KEY]);
  assert.deepEqual(store.touched, []);
  assert.deepEqual(stashed.touched, [], "dismissing keeps the stash");
  const later = await notice.findLegacyChats({ library: store, autosaves: stash([autosave, { ...autosave, key: "quality@900", modelVariantId: "quality", stashedAt: 900 }]) });
  assert.equal(notice.legacyNoticeShown(later, notice.readDismissedLegacyChats(local)), true);
  assert.equal(notice.legacyNoticeShown(chats, notice.readDismissedLegacyChats({ getItem: () => "not json" })), true);
  assert.equal(notice.legacyNoticeShown({ keys: [], conversationIds: [] }, new Set()), false);
});

test("downloads one JSON file with every old chat and autosave, as A2's export writes it", async () => {
  const saved = [];
  const store = library();
  const stashed = stash([autosave]);
  const file = await notice.downloadLegacyChats({ library: store, autosaves: stashed, now: Date.UTC(2026, 10, 1), save: (value) => saved.push(value) });
  assert.equal(saved.length, 1);
  assert.equal(saved[0], file);
  assert.equal(file.filename, "drowse-chats-before-6.0-2026-11-01.json");
  assert.equal(file.mimeType, "application/json");
  const document = JSON.parse(file.json);
  assert.equal(document.format, "drowse-legacy-conversations");
  assert.deepEqual(document.conversations.map((entry) => entry.id), ["old-a", "old-b"]);
  assert.deepEqual(document.autosaves.map((entry) => entry.model_variant_id), ["fastest"]);
  assert.deepEqual(store.touched, [], "downloading neither deletes nor closes a store it was given");
  assert.deepEqual(stashed.touched, [], "downloading keeps the stash");
  const none = await notice.downloadLegacyChats({ library: library([rows[1]]), autosaves: stash(), save: () => assert.fail("nothing to save") });
  assert.equal(none, null);
});

test("Your chats shows the notice and leaves old rows out of the unreadable list", async () => {
  const home = await readFile(new URL("../src/hosted/ui/HostedHome.svelte", import.meta.url), "utf8");
  assert.match(home, /from "\.\/legacyChats"/);
  assert.match(home, /legacyChats = await findLegacyChats\(\)/);
  assert.match(home, /refreshChats[\s\S]*?void refreshLegacyChats\(\)/, "the count follows the library");
  assert.match(home, /\{#if legacyShown\}\s*<section class="notice legacy-chats"/);
  assert.match(home, /<h2 id="legacy-chats-title">\{legacyChatsTitle\(legacyChats\.keys\.length\)\}<\/h2>/);
  assert.match(home, /onclick=\{dismissOldChats\}>Dismiss<\/button>/);
  assert.match(home, /onclick=\{\(\) => void downloadOldChats\(\)\}/);
  assert.match(home, /const unreadable = \$derived\(issues\.filter\(\(issue\) => !legacyIds\.has\(issue\.id\)\)\)/);
  assert.match(home, /\{#each unreadable as issue \(issue\.id\)\}/);
  assert.doesNotMatch(home, /\{#each issues as/);
  assert.doesNotMatch(home, /clearLegacyAutosaves/, "the notice never empties the stash");
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: HostedHome } = await server.ssrLoadModule("/src/hosted/ui/HostedHome.svelte");
  const snapshot = {
    phase: "supported", headline: "", detail: "", checks: [], models: [],
    download: { available: true, phase: "idle", reason: "" }, runtime: { available: true, phase: "unloaded", reason: "" },
  };
  const body = render(HostedHome, { props: { controller: { current: () => snapshot }, snapshot, onChooseModels() {}, onChats() {} } }).body;
  assert.doesNotMatch(body, /legacy-chats/, "no notice before the stores are read");
});
