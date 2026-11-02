// Saved chats from before drowse 6.0. The conversation snapshot is v8, and the
// library refuses an earlier one (exact-format rule), so a 5.x chat no longer
// opens. This is the data side of offering those chats as one downloaded file:
// find the stored rows whose snapshot predates v8 and serialize them as they
// are, without validating, migrating or opening them.
import {
  BrowserSavedConversationStore,
  type SavedConversationStore,
  type SavedConversationStoreRow,
} from "../../../../drowse/webui/src/lib/conversationLibrary";
import { CONVERSATION_SNAPSHOT_VERSION } from "../../../../drowse/webui/src/lib/conversationSnapshot";

export const LEGACY_CONVERSATIONS_FORMAT = "drowse-legacy-conversations";
export const LEGACY_CONVERSATIONS_FORMAT_VERSION = 1;

/** What the offer can say about one old chat without opening it. */
export interface LegacyConversationSummary {
  id: string;
  name: string | null;
  modelId: string | null;
  snapshotVersion: number;
  updatedAt: number | null;
}

/** The download: one JSON document holding every pre-v8 row verbatim. */
export interface LegacyConversationsFile {
  filename: string;
  mimeType: "application/json";
  json: string;
  conversations: LegacyConversationSummary[];
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** The snapshot version of a stored row, or null when it carries none. */
function snapshotVersion(value: unknown): number | null {
  if (!isObject(value) || !isObject(value.snapshot)) return null;
  const version = value.snapshot.version;
  return typeof version === "number" && Number.isSafeInteger(version) ? version : null;
}

/** Whether a stored saved-conversation row holds a snapshot from before v8 (a 5.x chat). */
export function isLegacyConversationRow(value: unknown): boolean {
  const version = snapshotVersion(value);
  return version !== null && version >= 1 && version < CONVERSATION_SNAPSHOT_VERSION;
}

function summary(row: SavedConversationStoreRow): LegacyConversationSummary {
  const value = row.value as Record<string, unknown>;
  const text = (field: unknown): string | null => (typeof field === "string" ? field : null);
  const time = (field: unknown): number | null =>
    typeof field === "number" && Number.isFinite(field) ? field : null;
  return {
    id: row.key,
    name: text(value.name),
    modelId: text(value.modelId),
    snapshotVersion: snapshotVersion(value)!,
    updatedAt: time(value.updatedAt),
  };
}

/** The library's pre-v8 rows, oldest first by `updatedAt`, untouched. */
export async function findLegacyConversations(
  store: SavedConversationStore = new BrowserSavedConversationStore(),
): Promise<SavedConversationStoreRow[]> {
  await store.initialize();
  const rows = await store.list();
  return rows
    .filter((row) => isLegacyConversationRow(row.value))
    .sort((a, b) => (summary(a).updatedAt ?? 0) - (summary(b).updatedAt ?? 0));
}

/** One JSON file with `rows` exactly as stored. */
export function legacyConversationsFile(
  rows: readonly SavedConversationStoreRow[],
  now = Date.now(),
): LegacyConversationsFile {
  const exportedAt = new Date(now).toISOString();
  const document = {
    format: LEGACY_CONVERSATIONS_FORMAT,
    format_version: LEGACY_CONVERSATIONS_FORMAT_VERSION,
    exported_at: exportedAt,
    note: "Saved chats from a drowse.ai build before 6.0. This version cannot open them; they are kept here as stored.",
    conversations: rows.map((row) => ({ id: row.key, record: row.value })),
  };
  return {
    filename: `drowse-chats-before-6.0-${exportedAt.slice(0, 10)}.json`,
    mimeType: "application/json",
    json: JSON.stringify(document, null, 2),
    conversations: rows.map(summary),
  };
}

/** The offer's one call: the file of every pre-v8 saved chat, or null when there is none. */
export async function exportLegacyConversations(
  store?: SavedConversationStore,
  now = Date.now(),
): Promise<LegacyConversationsFile | null> {
  const rows = await findLegacyConversations(store);
  return rows.length === 0 ? null : legacyConversationsFile(rows, now);
}
