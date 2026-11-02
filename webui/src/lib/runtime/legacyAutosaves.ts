// Hosted autosaves this version cannot open. The shared session persistence
// keeps one autosave per model variant and deletes a record that fails its
// validation (claim, save) or that a reset replaces. A 5.x autosave carries the
// same schemaVersion as a 6.0 one but 5.x rows, so 6.0 would delete it on the
// first load of that model. The worker therefore hands the persistence a store
// that copies any record which does not open as a 6.0 session into a separate
// stash before deleting it; the "download your old chats" offer exports the
// stash with the pre-v8 saved chats (legacyConversations.ts).
import {
  BrowserSessionStateStore,
  validatePersistedHostedSession,
  type HostedSessionStateStore,
  type PersistedHostedSession,
} from "../../../../drowse/webui/src/lib/sessionPersistence";

export const LEGACY_AUTOSAVE_DATABASE = "drowse-hosted-legacy-autosaves";
const DATABASE_VERSION = 1;
const AUTOSAVES_STORE = "autosaves";

/** One autosave copied verbatim before the session store deleted it. */
export interface StashedHostedAutosave {
  /** `${modelVariantId}@${stashedAt}`: a later stash never replaces an earlier one. */
  key: string;
  modelVariantId: string;
  stashedAt: number;
  record: unknown;
}

export interface HostedAutosaveStash {
  initialize(): Promise<void>;
  put(entry: StashedHostedAutosave): Promise<void>;
  list(): Promise<StashedHostedAutosave[]>;
  clear(): Promise<void>;
  close?(): void;
}

/** Whether `value`, stored under `modelVariantId`, opens as a session in this version. */
export function opensAsHostedSession(modelVariantId: string, value: unknown): boolean {
  try {
    validatePersistedHostedSession(value);
  } catch {
    return false;
  }
  return (value as PersistedHostedSession).modelVariantId === modelVariantId;
}

/**
 * The worker's session store: the shared IndexedDB store, plus a copy into the
 * stash of every record that does not open in this version before it is
 * deleted. A failed copy fails the delete, so nothing unopened is lost silently.
 * `clear` (the runtime's "clear all local data") empties both.
 */
export class LegacyStashingSessionStateStore implements HostedSessionStateStore {
  constructor(
    private readonly inner: HostedSessionStateStore = new BrowserSessionStateStore(),
    private readonly stash: HostedAutosaveStash = new BrowserHostedAutosaveStash(),
    private readonly now: () => number = Date.now,
  ) {}

  initialize(): Promise<void> {
    return this.inner.initialize();
  }

  read(modelVariantId: string): Promise<unknown | undefined> {
    return this.inner.read(modelVariantId);
  }

  write(record: PersistedHostedSession): Promise<void> {
    return this.inner.write(record);
  }

  async delete(modelVariantId: string): Promise<boolean> {
    const value = await this.inner.read(modelVariantId);
    if (value === undefined) return false;
    if (!opensAsHostedSession(modelVariantId, value)) {
      const stashedAt = this.now();
      await this.stash.initialize();
      await this.stash.put({
        key: `${modelVariantId}@${stashedAt}`,
        modelVariantId,
        stashedAt,
        record: value,
      });
    }
    return this.inner.delete(modelVariantId);
  }

  async clear(): Promise<void> {
    await this.inner.clear();
    await this.stash.initialize();
    await this.stash.clear();
  }

  close(): void {
    this.inner.close?.();
    this.stash.close?.();
  }
}

/** The stashed autosaves, oldest first. */
export async function findLegacyAutosaves(
  stash: HostedAutosaveStash = new BrowserHostedAutosaveStash(),
): Promise<StashedHostedAutosave[]> {
  await stash.initialize();
  const entries = await stash.list();
  return entries.sort((a, b) => a.stashedAt - b.stashedAt || a.key.localeCompare(b.key));
}

/** Drops every stashed autosave (after the person downloaded or declined them). */
export async function clearLegacyAutosaves(
  stash: HostedAutosaveStash = new BrowserHostedAutosaveStash(),
): Promise<void> {
  await stash.initialize();
  await stash.clear();
}

export class BrowserHostedAutosaveStash implements HostedAutosaveStash {
  private database: IDBDatabase | null = null;
  private opening: Promise<IDBDatabase> | null = null;

  async initialize(): Promise<void> {
    await this.open();
  }

  async put(entry: StashedHostedAutosave): Promise<void> {
    await this.run("readwrite", (store) => store.put(structuredClone(entry)));
  }

  async list(): Promise<StashedHostedAutosave[]> {
    return this.run("readonly", (store) => store.getAll() as IDBRequest<StashedHostedAutosave[]>);
  }

  async clear(): Promise<void> {
    await this.run("readwrite", (store) => store.clear());
  }

  close(): void {
    this.database?.close();
    this.database = null;
    this.opening = null;
  }

  private open(): Promise<IDBDatabase> {
    if (this.database !== null) return Promise.resolve(this.database);
    this.opening ??= new Promise<IDBDatabase>((resolve, reject) => {
      if (typeof indexedDB === "undefined") {
        reject(new Error("IndexedDB is unavailable"));
        return;
      }
      const request = indexedDB.open(LEGACY_AUTOSAVE_DATABASE, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(AUTOSAVES_STORE)) {
          request.result.createObjectStore(AUTOSAVES_STORE, { keyPath: "key" });
        }
      };
      request.onsuccess = () => {
        const database = request.result;
        database.onversionchange = () => this.close();
        this.database = database;
        resolve(database);
      };
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
      request.onblocked = () => reject(new Error("Another drowse tab is blocking the autosave stash"));
    }).catch((error) => {
      this.opening = null;
      throw error;
    });
    return this.opening;
  }

  private async run<T>(
    mode: IDBTransactionMode,
    operation: (store: IDBObjectStore) => IDBRequest<T>,
  ): Promise<T> {
    const database = await this.open();
    const transaction = database.transaction(AUTOSAVES_STORE, mode);
    const request = operation(transaction.objectStore(AUTOSAVES_STORE));
    return new Promise<T>((resolve, reject) => {
      transaction.oncomplete = () => resolve(request.result);
      transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
      transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
    });
  }
}
