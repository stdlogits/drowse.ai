/** Which model the hosted entry flow last finished onboarding with, so a
 *  returning visitor lands on their chats instead of the model picker. */
import { migrateLegacyStorageItem, removeLegacyStorageItem } from "../../../../drowse/webui/src/lib/brandMigration";

const ENTRY_MEMORY_KEY = "drowse.entry.v1";

export interface EntryMemory {
  version: 1;
  completedAt: number;
  lastModelVariantId: string;
}

type WebStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function readEntryMemory(
  storage: WebStorage | undefined = browserStorage("localStorage"),
): EntryMemory | null {
  if (!storage) return null;
  try {
    const raw = migrateLegacyStorageItem(storage, ENTRY_MEMORY_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<EntryMemory>;
    if (
      value.version !== 1 ||
      !Number.isSafeInteger(value.completedAt) ||
      (value.completedAt ?? -1) < 0 ||
      typeof value.lastModelVariantId !== "string" ||
      !value.lastModelVariantId.trim()
    ) return null;
    return value as EntryMemory;
  } catch {
    return null;
  }
}

export function rememberCompletedOnboarding(
  modelVariantId: string,
  storage: WebStorage | undefined = browserStorage("localStorage"),
  now = Date.now(),
): void {
  if (!storage || !modelVariantId.trim()) return;
  try {
    storage.setItem(ENTRY_MEMORY_KEY, JSON.stringify({
      version: 1,
      completedAt: now,
      lastModelVariantId: modelVariantId,
    } satisfies EntryMemory));
  } catch {}
}

/**
 * Called after a model is removed from this device. Points entry memory at
 * another installed model, or clears it when none remain, so the chats page
 * doesn't keep asking for a model that was deleted on purpose.
 */
export function forgetEntryModel(
  removedModelVariantId: string,
  replacementModelVariantId: string | null,
  storage: WebStorage | undefined = browserStorage("localStorage"),
): void {
  const memory = readEntryMemory(storage);
  if (!storage || memory?.lastModelVariantId !== removedModelVariantId) return;
  try {
    if (replacementModelVariantId?.trim()) {
      storage.setItem(ENTRY_MEMORY_KEY, JSON.stringify({
        ...memory,
        lastModelVariantId: replacementModelVariantId,
      } satisfies EntryMemory));
    } else {
      removeLegacyStorageItem(storage, ENTRY_MEMORY_KEY);
    }
  } catch {}
}

function browserStorage(name: "localStorage" | "sessionStorage"): Storage | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    return window[name];
  } catch {
    return undefined;
  }
}
