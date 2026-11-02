// The notice in Your chats for chats saved before Drowse 6.0. The library refuses a snapshot older
// than v8, and the worker stashes a 5.x autosave instead of deleting it (lib/runtime/legacyAutosaves.ts),
// so neither opens here. The notice counts both, downloads them as one file (A2's
// exportLegacyConversations, which keeps every record as stored) and can be dismissed on this device.
// Dismissing deletes nothing; clearing all local data empties the stash and the library, and this
// key with them (drowse's clearDrowseLocalData clears every `drowse.` key).
import {
  exportLegacyConversations,
  findLegacyConversations,
  type LegacyConversationsFile,
} from "../../lib/runtime/legacyConversations";
import {
  BrowserHostedAutosaveStash,
  findLegacyAutosaves,
  type HostedAutosaveStash,
} from "../../lib/runtime/legacyAutosaves";
import {
  BrowserSavedConversationStore,
  type SavedConversationStore,
} from "../../../../drowse/webui/src/lib/conversationLibrary";

export const LEGACY_CHATS_DISMISSED_KEY = "drowse.legacy-chats-dismissed.v1";

/** What the notice knows without opening anything: one key per old chat or stashed autosave. */
export interface LegacyChats {
  /** `chat:<id>` for each pre-v8 saved chat, `autosave:<key>` for each stashed autosave. */
  keys: string[];
  /** The library ids of the old saved chats, which the list leaves to the notice. */
  conversationIds: string[];
}

export const NO_LEGACY_CHATS: LegacyChats = { keys: [], conversationIds: [] };

interface Stores {
  library?: SavedConversationStore;
  autosaves?: HostedAutosaveStash;
}

type StorageLike = Pick<Storage, "getItem" | "setItem">;

function browserStorage(): StorageLike | null {
  try { return globalThis.localStorage ?? null; } catch { return null; }
}

/** Runs `read` over the given stores, or over browser stores it opens and closes again. */
async function withStores<T>(stores: Stores, read: (library: SavedConversationStore, autosaves: HostedAutosaveStash) => Promise<T>): Promise<T> {
  const library = stores.library ?? new BrowserSavedConversationStore();
  const autosaves = stores.autosaves ?? new BrowserHostedAutosaveStash();
  try {
    return await read(library, autosaves);
  } finally {
    if (!stores.library) library.close?.();
    if (!stores.autosaves) autosaves.close?.();
  }
}

export async function findLegacyChats(stores: Stores = {}): Promise<LegacyChats> {
  return withStores(stores, async (library, autosaves) => {
    const conversationIds = (await findLegacyConversations(library)).map((row) => row.key);
    const stashed = await findLegacyAutosaves(autosaves);
    return {
      keys: [...conversationIds.map((id) => `chat:${id}`), ...stashed.map((entry) => `autosave:${entry.key}`)],
      conversationIds,
    };
  });
}

export function readDismissedLegacyChats(storage: StorageLike | null = browserStorage()): Set<string> {
  try {
    const value: unknown = JSON.parse(storage?.getItem(LEGACY_CHATS_DISMISSED_KEY) ?? "[]");
    return new Set(Array.isArray(value) ? value.filter((key): key is string => typeof key === "string") : []);
  } catch {
    return new Set();
  }
}

/** Shown while any old chat or autosave is one the person has not dismissed on this device. */
export function legacyNoticeShown(chats: LegacyChats, dismissed: ReadonlySet<string>): boolean {
  return chats.keys.some((key) => !dismissed.has(key));
}

/** Hides the notice for these chats on this device. A chat stashed later brings it back. */
export function dismissLegacyChats(chats: LegacyChats, storage: StorageLike | null = browserStorage()): Set<string> {
  const dismissed = new Set([...readDismissedLegacyChats(storage), ...chats.keys]);
  try { storage?.setItem(LEGACY_CHATS_DISMISSED_KEY, JSON.stringify([...dismissed])); } catch { /* Hidden for this visit. */ }
  return dismissed;
}

export function legacyChatsTitle(count: number): string {
  return `${count === 1 ? "1 chat" : `${count} chats`} from before Drowse 6.0 can’t be opened here`;
}

function saveFile(file: LegacyConversationsFile): void {
  const url = URL.createObjectURL(new Blob([file.json], { type: file.mimeType }));
  const link = document.createElement("a");
  link.href = url;
  link.download = file.filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

/** Downloads every old chat and stashed autosave as one JSON file; null when there is none. */
export async function downloadLegacyChats(
  options: Stores & { now?: number; save?: (file: LegacyConversationsFile) => void } = {},
): Promise<LegacyConversationsFile | null> {
  const file = await withStores(options, (library, autosaves) => exportLegacyConversations({ library, autosaves, now: options.now }));
  if (file) (options.save ?? saveFile)(file);
  return file;
}
