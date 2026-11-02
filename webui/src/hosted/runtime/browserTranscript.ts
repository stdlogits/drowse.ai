import { parseDocument, stringify } from "yaml";
import type {
  FinishReason,
  LoomNodeJSON,
  LoomTreeJSON,
  RecipeJSON,
  SessionInfo,
} from "../../../../drowse/webui/src/lib/types";
import { FINISH_REASONS } from "../../../../drowse/webui/src/lib/types";

// drowse's `loom/transcript.py` v3: the cast model plus each generated turn's finish reason.
const TRANSCRIPT_VERSION = 3;
const ROOT_KEYS = new Set(["drowse_transcript", "turns", "model_id", "system_prompt", "cast", "probes"]);
const TURN_KEYS = new Set(["role", "text", "speaker", "thinking", "recipe", "readings", "finish_reason"]);
const MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024;
const MAX_TURNS = 100_000;

export interface BrowserTranscriptTurn {
  role: "user" | "assistant" | "system";
  text: string;
  speaker: string | null;
  thinking: string | null;
  recipe: RecipeJSON | null;
  readings: Record<string, number>;
  /** A generated turn's finish reason (`cancelled`: cut short by a Stop or a disconnect). */
  finishReason: FinishReason | null;
}

export interface BrowserTranscript {
  modelId: string | null;
  systemPrompt: string | null;
  probes: Array<{ name: string; sha256: string }>;
  turns: BrowserTranscriptTurn[];
  cast: LoomTreeJSON["cast"];
}

export function exportBrowserTranscript(
  tree: LoomTreeJSON,
  session: SessionInfo,
  nodeId: string,
  probeHashes: Readonly<Record<string, string>> = {},
): string {
  const turns = transcriptPath(tree, nodeId)
    .filter((node) => node.id !== tree.root_id)
    .map((node) => {
      const turn: Record<string, unknown> = { role: node.role, text: node.text };
      if (node.role_label !== null) turn.speaker = node.role_label;
      if (node.thinking_text !== null) turn.thinking = node.thinking_text;
      if (node.recipe !== null) turn.recipe = structuredClone(node.recipe);
      if (Object.keys(node.aggregate_readings).length > 0) {
        turn.readings = structuredClone(node.aggregate_readings);
      }
      if (node.recipe !== null && node.finish_reason !== null) turn.finish_reason = node.finish_reason;
      return turn;
    });
  const document: Record<string, unknown> = {
    drowse_transcript: TRANSCRIPT_VERSION,
    model_id: tree.model_id ?? session.model_id,
    system_prompt: session.config.system_prompt,
  };
  if (Object.keys(tree.cast).length > 0) document.cast = structuredClone(tree.cast);
  document.probes = session.probes.map((name) => ({
    name,
    sha256: probeHashes[name] ?? "",
  }));
  document.turns = turns;
  return stringify(document, { lineWidth: 0 });
}

export function parseBrowserTranscript(source: string): BrowserTranscript {
  if (typeof source !== "string" || source.includes("\0")) {
    throw transcriptError("INVALID_TRANSCRIPT", "Transcript input must be text");
  }
  if (new TextEncoder().encode(source).byteLength > MAX_TRANSCRIPT_BYTES) {
    throw transcriptError("INVALID_TRANSCRIPT", "Transcript exceeds the 16 MiB browser limit");
  }
  let value: unknown;
  try {
    const document = parseDocument(source, {
      prettyErrors: false,
      schema: "core",
      uniqueKeys: true,
    });
    if (document.errors.length > 0) throw document.errors[0];
    value = document.toJS({ maxAliasCount: 0 });
  } catch (error) {
    throw transcriptError(
      "INVALID_TRANSCRIPT",
      `Transcript YAML is invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const root = record(value, "transcript root");
  const unknownRoot = Object.keys(root).filter((key) => !ROOT_KEYS.has(key));
  if (unknownRoot.length > 0) {
    throw transcriptError("INVALID_TRANSCRIPT", `Transcript has unknown fields: ${unknownRoot.sort().join(", ")}`);
  }
  if (root.drowse_transcript !== TRANSCRIPT_VERSION) {
    throw transcriptError(
      "INVALID_TRANSCRIPT",
      `Transcript version ${String(root.drowse_transcript)} is unsupported; expected ${TRANSCRIPT_VERSION}`,
    );
  }
  const modelId = nullableString(root.model_id, "model_id", 1024);
  const systemPrompt = nullableString(root.system_prompt, "system_prompt", 4 * 1024 * 1024);
  requireKeys(root, ["turns"], "transcript");
  const rawProbes = array(root.probes ?? [], "probes", 16_384);
  const probes = rawProbes.map((item, index) => {
    const probe = record(item, `probes[${index}]`);
    requireKeys(probe, ["name", "sha256"], `probes[${index}]`);
    const unknown = Object.keys(probe).filter((key) => key !== "name" && key !== "sha256");
    if (unknown.length > 0) {
      throw transcriptError("INVALID_TRANSCRIPT", `probes[${index}] has unknown fields: ${unknown.sort().join(", ")}`);
    }
    return {
      name: requiredString(probe.name, `probes[${index}].name`, 1024),
      sha256: requiredString(probe.sha256, `probes[${index}].sha256`, 128),
    };
  });
  const rawTurns = array(root.turns, "turns", MAX_TURNS);
  const turns = rawTurns.map((item, index) => parseTurn(item, index));
  const cast = parseCast(root.cast ?? {});
  return { modelId, systemPrompt, probes, turns, cast };
}

function parseTurn(value: unknown, index: number): BrowserTranscriptTurn {
  const turn = record(value, `turns[${index}]`);
  const unknown = Object.keys(turn).filter((key) => !TURN_KEYS.has(key));
  if (unknown.length > 0) {
    throw transcriptError("INVALID_TRANSCRIPT", `turns[${index}] has unknown fields: ${unknown.sort().join(", ")}`);
  }
  requireKeys(turn, ["role", "text"], `turns[${index}]`);
  const role = turn.role;
  if (role !== "user" && role !== "assistant" && role !== "system") {
    throw transcriptError("INVALID_TRANSCRIPT", `turns[${index}].role is invalid`);
  }
  const speaker = nullableString(turn.speaker, `turns[${index}].speaker`, 256);
  if (speaker !== null && !/^[a-z0-9._-]+$/u.test(speaker)) {
    throw transcriptError("INVALID_TRANSCRIPT", `turns[${index}].speaker is not a role slug`);
  }
  const rawReadings = record(turn.readings ?? {}, `turns[${index}].readings`);
  const readings: Record<string, number> = {};
  for (const [name, reading] of Object.entries(rawReadings)) {
    if (typeof reading !== "number" || !Number.isFinite(reading)) {
      throw transcriptError("INVALID_TRANSCRIPT", `turns[${index}].readings.${name} is invalid`);
    }
    readings[name] = reading;
  }
  const recipe = turn.recipe === null || turn.recipe === undefined
    ? null
    : structuredClone(record(turn.recipe, `turns[${index}].recipe`)) as unknown as RecipeJSON;
  const finishReason = turn.finish_reason ?? null;
  if (finishReason !== null) {
    if (!(FINISH_REASONS as readonly unknown[]).includes(finishReason)) {
      throw transcriptError(
        "INVALID_TRANSCRIPT",
        `turns[${index}].finish_reason must be one of ${[...FINISH_REASONS].sort().join(", ")}`,
      );
    }
    if (recipe === null) {
      throw transcriptError(
        "INVALID_TRANSCRIPT",
        `turns[${index}].finish_reason needs a recipe: only a generated turn has one`,
      );
    }
  }
  return {
    role,
    text: requiredString(turn.text, `turns[${index}].text`, 4 * 1024 * 1024),
    speaker,
    thinking: nullableString(turn.thinking, `turns[${index}].thinking`, 4 * 1024 * 1024),
    recipe,
    readings,
    finishReason: finishReason as FinishReason | null,
  };
}

function parseCast(value: unknown): LoomTreeJSON["cast"] {
  const source = record(value, "cast");
  const cast: LoomTreeJSON["cast"] = {};
  for (const [label, memberValue] of Object.entries(source)) {
    if (!/^[a-z0-9._-]+$/u.test(label)) {
      throw transcriptError("INVALID_TRANSCRIPT", `Cast label ${label} is invalid`);
    }
    const member = record(memberValue, `cast.${label}`);
    const keys = Object.keys(member);
    if (keys.some((key) => key !== "recipe" && key !== "notes")) {
      throw transcriptError("INVALID_TRANSCRIPT", `cast.${label} contains unknown fields`);
    }
    cast[label] = {
      recipe: member.recipe === null || member.recipe === undefined
        ? null
        : structuredClone(record(member.recipe, `cast.${label}.recipe`)) as unknown as RecipeJSON,
      notes: requiredString(member.notes ?? "", `cast.${label}.notes`, 256 * 1024),
    };
  }
  return cast;
}

function transcriptPath(tree: LoomTreeJSON, nodeId: string): LoomNodeJSON[] {
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  const path: LoomNodeJSON[] = [];
  const visited = new Set<string>();
  let node = byId.get(nodeId);
  if (!node) throw transcriptError("LOOM_NODE_NOT_FOUND", `Conversation node ${nodeId} does not exist`);
  while (node) {
    if (visited.has(node.id)) throw transcriptError("INVALID_TRANSCRIPT", "Conversation path contains a cycle");
    visited.add(node.id);
    path.push(node);
    node = node.parent_id === null ? undefined : byId.get(node.parent_id);
  }
  path.reverse();
  return path;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw transcriptError("INVALID_TRANSCRIPT", `${label} must be a mapping`);
  }
  return value as Record<string, unknown>;
}

/** drowse's `_check_keys` required half: every listed key is present. */
function requireKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  const missing = keys.filter((key) => !(key in value)).sort();
  if (missing.length > 0) {
    throw transcriptError("INVALID_TRANSCRIPT", `${label} is missing: ${missing.join(", ")}`);
  }
}

function array(value: unknown, label: string, maximum: number): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) {
    throw transcriptError("INVALID_TRANSCRIPT", `${label} must be a list with at most ${maximum} entries`);
  }
  return value;
}

function nullableString(value: unknown, label: string, maximum: number): string | null {
  if (value === null || value === undefined) return null;
  return requiredString(value, label, maximum);
}

function requiredString(value: unknown, label: string, maximum: number): string {
  if (typeof value !== "string" || value.length > maximum || value.includes("\0")) {
    throw transcriptError("INVALID_TRANSCRIPT", `${label} must be valid text`);
  }
  return value;
}

function transcriptError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 400, recoverable: true });
}
