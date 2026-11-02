// Saved chats from before drowse 6.0. The conversation snapshot is v8, and the
// library refuses an earlier one (exact-format rule), so a 5.x chat no longer
// opens. This is the data side of offering those chats as one downloaded file:
// find the stored rows whose snapshot predates v8, and the hosted autosaves the
// worker stashed instead of deleting (legacyAutosaves.ts), and serialize them as
// they are, without validating, migrating or opening them.
import {
  findLegacyAutosaves,
  type HostedAutosaveStash,
  type StashedHostedAutosave,
} from "./legacyAutosaves";
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

/** What the offer can say about one stashed autosave. */
export interface LegacyAutosaveSummary {
  modelVariantId: string;
  stashedAt: number;
  updatedAt: number | null;
}

/** The download: one JSON document holding every pre-v8 row and stashed autosave verbatim. */
export interface LegacyConversationsFile {
  filename: string;
  mimeType: "application/json";
  json: string;
  conversations: LegacyConversationSummary[];
  autosaves: LegacyAutosaveSummary[];
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

function autosaveSummary(entry: StashedHostedAutosave): LegacyAutosaveSummary {
  const updatedAt = isObject(entry.record) ? entry.record.updatedAt : null;
  return {
    modelVariantId: entry.modelVariantId,
    stashedAt: entry.stashedAt,
    updatedAt: typeof updatedAt === "number" && Number.isFinite(updatedAt) ? updatedAt : null,
  };
}

/** One JSON file with `rows` and `autosaves` exactly as stored. */
export function legacyConversationsFile(
  rows: readonly SavedConversationStoreRow[],
  autosaves: readonly StashedHostedAutosave[] = [],
  now = Date.now(),
): LegacyConversationsFile {
  const exportedAt = new Date(now).toISOString();
  const document = {
    format: LEGACY_CONVERSATIONS_FORMAT,
    format_version: LEGACY_CONVERSATIONS_FORMAT_VERSION,
    exported_at: exportedAt,
    note: "Saved chats and autosaves from a drowse.ai build before 6.0. This version cannot open them; they are kept here as stored.",
    conversations: rows.map((row) => ({ id: row.key, record: row.value })),
    autosaves: autosaves.map((entry) => ({
      model_variant_id: entry.modelVariantId,
      stashed_at: new Date(entry.stashedAt).toISOString(),
      record: entry.record,
    })),
  };
  return {
    filename: `drowse-chats-before-6.0-${exportedAt.slice(0, 10)}.json`,
    mimeType: "application/json",
    json: JSON.stringify(document, null, 2),
    conversations: rows.map(summary),
    autosaves: autosaves.map(autosaveSummary),
  };
}

/**
 * The offer's one call: the file of every pre-v8 saved chat and stashed
 * autosave, or null when there is none.
 */
export async function exportLegacyConversations(
  options: { library?: SavedConversationStore; autosaves?: HostedAutosaveStash; now?: number } = {},
): Promise<LegacyConversationsFile | null> {
  const rows = await findLegacyConversations(options.library);
  const autosaves = await findLegacyAutosaves(options.autosaves);
  return rows.length === 0 && autosaves.length === 0
    ? null
    : legacyConversationsFile(rows, autosaves, options.now ?? Date.now());
}
