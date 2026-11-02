// The steering grammar: a TypeScript mirror of drowse/steering/grammar.py (lexer, parser, fold,
// format, term parts), drowse/model/layers.py (the layer-selector language of a `{…}` clause)
// and composer.select_term_layers. Python is the reference; the goldens
// drowse/tests/fixtures/{steering_ir,recipe_chips,browser_steering_expression}.json pin both.
//
// `parseSteeringExpression` is Python's `_Parser.parse`: the syntax tree, before any resolution.
// `parseSteering` is `parse_expr`: the tree folded into the IR (`Steering.alphas`, in order),
// bare atoms resolved against the installed manifolds. `formatSteering` is `format_expr` and
// `steeringParts` is `term_parts` on the wire (`SteeringPartJSON`).

import { formatG } from "../../../../drowse/webui/src/lib/expression";
import { formatLayerIndices } from "../../../../drowse/webui/src/lib/steering/layers";
import type { SteeringPartJSON } from "../../../../drowse/webui/src/lib/types";

export const DEFAULT_STEERING_COEFFICIENT = 0.5;
export const DEFAULT_ABLATION_COEFFICIENT = 1;

export type SteeringProjectionOperator = "~" | "|";
export type SteeringComparisonOperator = ">" | ">=" | "<" | "<=";

export type SteeringPhase =
  | { kind: "both" }
  | { kind: "after_thinking" }
  | { kind: "prompt_only" }
  | { kind: "thinking_only" }
  | { kind: "generated_only" }
  | { kind: "first"; tokens: number }
  | { kind: "after"; tokens: number };

export interface SteeringProbeGate {
  probe: string;
  operator: SteeringComparisonOperator;
  threshold: number;
}

/** A trigger: its window (`phase`) and an optional probe gate. A bare `@when:` is the
 *  `generated_only` window with a gate, as Python's `Trigger.when`. */
export interface SteeringTrigger {
  phase: SteeringPhase;
  gate: SteeringProbeGate | null;
}

export interface SteeringAtom {
  namespace: string | null;
  /** The name: verbatim for a concept or reserved instrument atom, canonical (`L8n344`) for a unit. */
  concept: string;
  variant: string;
  column: number;
}

export interface SteeringSelector {
  base: SteeringAtom;
  projection: SteeringProjectionOperator | null;
  onto: SteeringAtom | null;
  manifoldPosition: readonly number[] | string | null;
}

/** A `{…}` clause as written: `"all"`, `"workspace"` or the sorted indices; `null` without one. */
export type WrittenLayerClause = readonly number[] | "all" | "workspace" | null;

/** A term's layer scope in the IR (grammar `LayerScope`): `null` acts on every fitted layer,
 *  `"workspace"` on the workspace band within them, an index list on exactly those. */
export type LayerScope = readonly number[] | "workspace" | null;

export interface SteeringExpressionTerm {
  /** The comma run, each value carrying the term's sign (`-0.6,0.3` is `[-0.6, -0.3]`). */
  coefficients: readonly number[];
  selector: SteeringSelector;
  trigger: SteeringTrigger | null;
  ablation: boolean;
  layers: WrittenLayerClause;
}

export interface SteeringExpression {
  terms: readonly SteeringExpressionTerm[];
}

/** Namespaces that name instrument units rather than fitted manifolds
 *  (drowse/naming.py `RESERVED_NAMESPACES`): no manifold or pack may live
 *  under one, and a steering atom in one keeps its name verbatim and takes
 *  no variant suffix. */
export const RESERVED_NAMESPACES: ReadonlySet<string> = new Set([
  "jlens", "slens", "sae", "neuron", "head",
]);
/** The reserved namespaces whose names are model units (`naming.UNIT_NAMESPACES`). */
export const UNIT_NAMESPACES: ReadonlySet<string> = new Set(["neuron", "head"]);
/** Lens namespaces whose units may also be named by an exact vocab id, `#<id>`. */
const TOKEN_ID_NAMESPACES: ReadonlySet<string> = new Set(["jlens", "slens"]);

/** A malformed expression (Python `SteeringExprError`, 400). `code` is the Python class name. */
export class SteeringExpressionError extends Error {
  readonly code: string = "SteeringExprError";

  constructor(message: string, readonly column: number | null = null) {
    super(column === null ? message : `${message} (col ${column})`);
    this.name = "SteeringExpressionError";
  }
}

/** An atom that does not resolve (Python `store.selectors.SelectorError`, 400). */
export class SteeringSelectorError extends Error {
  readonly code: string = "SelectorError";

  constructor(message: string) {
    super(message);
    this.name = "SteeringSelectorError";
  }
}

/** A bare atom that names more than one installed manifold (`AmbiguousSelectorError`). Its
 *  message is the one drowse shows a person (`user_message`), which adds the tip. */
export class AmbiguousSteeringSelectorError extends SteeringSelectorError {
  override readonly code: string = "AmbiguousSelectorError";

  constructor(message: string) {
    super(message.includes("namespace/name")
      ? message
      : `${message} (disambiguate with namespace/name)`);
    this.name = "AmbiguousSteeringSelectorError";
  }
}

// ------------------------------------------------------------------ lexer ---

type TokenKind =
  | "AMP" | "AT" | "BANG" | "COLON" | "COMMA" | "DOT" | "EOF" | "GT" | "GTE" | "HASH"
  | "IDENT" | "LAYERS" | "LBRACK" | "LT" | "LTE" | "MINUS" | "NUM" | "ORTHO" | "PERCENT"
  | "PLUS" | "RBRACK" | "SLASH" | "STAR" | "TILDE";

interface Token {
  kind: TokenKind;
  value: string | number;
  column: number;
}

const SYMBOLS: Record<string, TokenKind> = {
  ".": "DOT", "/": "SLASH", ":": "COLON", "*": "STAR", "+": "PLUS", "-": "MINUS", "@": "AT",
  "~": "TILDE", "|": "ORTHO", "!": "BANG", "%": "PERCENT", ",": "COMMA", "[": "LBRACK",
  "]": "RBRACK", "&": "AMP", "#": "HASH",
};

// JavaScript's `\d` is ASCII 0-9, as the grammar requires of every digit.
const NUMBER = /^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/u;
const DIGIT = /^[0-9]$/u;
const IDENT_START = /^[A-Za-z]$/u;
const IDENT_CHAR = /^[A-Za-z0-9_]$/u;

/** `str.isspace`: Unicode White_Space without U+FEFF, plus the four information separators
 *  and NEL, which JavaScript's `\s` misses. */
const PYTHON_SPACE = /^[\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]$/u;

function isPythonSpace(character: string): boolean {
  return PYTHON_SPACE.test(character);
}

/** Python's `str.strip()` (no argument). */
export function pythonStrip(text: string): string {
  return pythonRstrip(pythonLstrip(text));
}

function pythonLstrip(text: string): string {
  const characters = Array.from(text);
  let start = 0;
  while (start < characters.length && isPythonSpace(characters[start])) start += 1;
  return characters.slice(start).join("");
}

function pythonRstrip(text: string): string {
  const characters = Array.from(text);
  let end = characters.length;
  while (end > 0 && isPythonSpace(characters[end - 1])) end -= 1;
  return characters.slice(0, end).join("");
}

function codePointLength(text: string): number {
  let length = 0;
  for (const _ of text) length += 1;
  return length;
}

/** `_lex`: tokens over the text's code points (Python indexes a `str` by code point), so every
 *  column agrees with drowse's even past an astral character. */
function lex(source: string): Token[] {
  const text = Array.from(source);
  const tokens: Token[] = [];
  let index = 0;
  while (index < text.length) {
    const character = text[index];
    if (isPythonSpace(character)) {
      index += 1;
      continue;
    }
    if (DIGIT.test(character) || (character === "." && DIGIT.test(text[index + 1] ?? ""))) {
      const match = NUMBER.exec(text.slice(index).join(""))!;
      const value = Number(match[0]);
      if (!Number.isFinite(value)) throw new SteeringExpressionError("number must be finite", index);
      tokens.push({ kind: "NUM", value, column: index });
      index += match[0].length;
      continue;
    }
    if (character === ">" || character === "<") {
      const equals = text[index + 1] === "=";
      tokens.push({
        kind: character === ">" ? (equals ? "GTE" : "GT") : (equals ? "LTE" : "LT"),
        value: equals ? `${character}=` : character,
        column: index,
      });
      index += equals ? 2 : 1;
      continue;
    }
    const symbol = SYMBOLS[character];
    if (symbol !== undefined) {
      tokens.push({ kind: symbol, value: character, column: index });
      index += 1;
      continue;
    }
    if (character === "{") {
      // A layer clause: its body is the layer-selector language, handed over verbatim.
      const close = text.indexOf("}", index + 1);
      if (close < 0) {
        throw new SteeringExpressionError("unterminated layer clause; close it with '}'", index);
      }
      tokens.push({ kind: "LAYERS", value: text.slice(index + 1, close).join(""), column: index });
      index = close + 1;
      continue;
    }
    if (IDENT_START.test(character)) {
      const start = index++;
      while (index < text.length) {
        const next = text[index];
        if (IDENT_CHAR.test(next)) {
          index += 1;
          continue;
        }
        // A dash between two identifier characters joins segments (`sae-release`).
        if (next === "-" && IDENT_CHAR.test(text[index + 1] ?? "")) {
          index += 1;
          continue;
        }
        break;
      }
      tokens.push({ kind: "IDENT", value: text.slice(start, index).join(""), column: start });
      continue;
    }
    if (character === '"' || character === "'") {
      throw new SteeringExpressionError(
        "quoted identifiers are not supported; use underscores for multi-word concept names " +
          "(e.g. 'artificial_intelligence') and '.' for bipolar pairs (e.g. " +
          "'human.artificial_intelligence')",
        index,
      );
    }
    throw new SteeringExpressionError(`unexpected character ${pythonStringRepr(character)}`, index);
  }
  tokens.push({ kind: "EOF", value: "", column: text.length });
  return tokens;
}

// ---------------------------------------------------------- layer selector ---

const LAYER_ENTRY = /^([0-9]+)[ \t\n\r\f\v]*(?:-[ \t\n\r\f\v]*([0-9]+))?$/u;

class LayerSelectorError extends Error {
  constructor(message: string, readonly offset: number | null) {
    super(message);
  }
}

/** `model/layers.py::parse_layer_selector(text, allow_sample=False)`: `"4-6,11"` → `[4, 5, 6, 11]`
 *  in written order; `all` / `workspace` pass through lower-cased; blank is `null`. */
function parseLayerSelector(text: string): number[] | "all" | "workspace" | null {
  // Offsets count code points and whitespace is Python's `str.isspace`, as `parse_layer_selector`
  // sees the clause body.
  const stripped = pythonStrip(text);
  if (!stripped) return null;
  const lowered = stripped.toLowerCase();
  if (lowered === "all" || lowered === "workspace") return lowered;
  if (lowered === "sample") {
    throw new LayerSelectorError(
      `unknown layer mode ${pythonStringRepr(stripped)} (want all, workspace, or an index list)`,
      codePointLength(text.slice(0, text.indexOf(stripped))),
    );
  }
  const layers: number[] = [];
  const seen = new Set<number>();
  let start = 0;
  for (const part of text.split(",")) {
    const body = pythonStrip(part);
    const offset = start + codePointLength(body ? part.slice(0, part.indexOf(body)) : part);
    start += codePointLength(part) + 1;
    if (!body) throw new LayerSelectorError(`empty entry in layers ${pythonStringRepr(text)}`, offset);
    const match = LAYER_ENTRY.exec(body);
    if (match === null) {
      if (body.startsWith("-") && LAYER_ENTRY.test(pythonLstrip(body.slice(1)))) {
        throw new LayerSelectorError(
          `layer indices are non-negative; got ${pythonStringRepr(body)}`,
          offset,
        );
      }
      throw new LayerSelectorError(
        `malformed layers ${pythonStringRepr(text)} (want comma-separated indices or a-b ranges, ` +
          "or one of all, workspace)",
        offset,
      );
    }
    const low = Number.parseInt(match[1], 10);
    const high = match[2] === undefined ? low : Number.parseInt(match[2], 10);
    if (high < low) {
      throw new LayerSelectorError(
        `reversed layer range ${pythonStringRepr(body)} (write ${high}-${low})`,
        offset,
      );
    }
    for (let layer = low; layer <= high; layer += 1) {
      if (seen.has(layer)) {
        throw new LayerSelectorError(`layer ${layer} is named twice in ${pythonStringRepr(text)}`, offset);
      }
      seen.add(layer);
      layers.push(layer);
    }
  }
  return layers;
}

/** The canonical `{…}` clause of a scope (`format_layer_clause`); `""` for every fitted layer. */
export function formatLayerClause(layers: LayerScope): string {
  if (layers === null) return "";
  if (layers === "workspace") return "{workspace}";
  return `{${formatLayerIndices(layers)}}`;
}

/** The workspace band (`model/arch.py::workspace_layer_indices`): 40–90% of the depth, or every
 *  candidate when none falls inside. */
export function workspaceLayerIndices(candidates: Iterable<number>, nLayers: number): number[] {
  const available = [...candidates];
  const band = available.filter((layer) => {
    const depth = layer / Math.max(nLayers - 1, 1);
    return depth >= 0.4 && depth <= 0.9;
  });
  return band.length > 0 ? band : available;
}

/** The layers a term acts on (`composer.select_term_layers`), `null` for every fitted one. An
 *  index scope must lie inside `fitted`; `workspace` intersects the band with it. Either failing
 *  raises the Python message (its `SteeringCompositionError`, 422). */
export function selectTermLayers(
  term: string,
  fitted: Iterable<number>,
  scope: LayerScope,
  nLayers: number,
): number[] | null {
  if (scope === null) return null;
  const have = [...new Set(fitted)].sort((a, b) => a - b);
  const haveText = have.length > 0 ? formatLayerIndices(have) : "none";
  if (scope === "workspace") {
    const band = workspaceLayerIndices(Array.from({ length: nLayers }, (_, index) => index), nLayers);
    const chosen = band.filter((layer) => have.includes(layer)).sort((a, b) => a - b);
    if (chosen.length === 0) {
      throw new SteeringCompositionError(
        `steering term '${term}' selects the workspace band (layers ` +
          `${band.length > 0 ? formatLayerIndices(band) : "none"}) but is fitted at layers ${haveText}`,
      );
    }
    return chosen;
  }
  const missing = [...new Set(scope)].filter((layer) => !have.includes(layer)).sort((a, b) => a - b);
  if (missing.length > 0) {
    throw new SteeringCompositionError(
      `steering term '${term}' selects layer(s) ${formatLayerIndices(missing)} it is not fitted ` +
        `at; fitted layers: ${haveText}`,
    );
  }
  return [...scope];
}

/** A term that cannot be composed as written (Python `SteeringCompositionError`, 422). */
export class SteeringCompositionError extends Error {
  readonly code: string = "SteeringCompositionError";
  readonly status = 422;

  constructor(message: string) {
    super(message);
    this.name = "SteeringCompositionError";
  }
}

// ----------------------------------------------------------------- parser ---

const PHASES: Record<string, SteeringPhase> = {
  both: { kind: "both" },
  after: { kind: "after_thinking" },
  before: { kind: "prompt_only" },
  thinking: { kind: "thinking_only" },
  response: { kind: "generated_only" },
  prompt: { kind: "prompt_only" },
  generated: { kind: "generated_only" },
};

const TRIGGER_FORMS_HELP =
  "valid: after, before, both, generated, prompt, response, thinking; 'first:<n>' / " +
  "'after:<n>' token windows; 'when:<probe><op><threshold>' probe gates; or a compound " +
  "'<preset>&when:<probe><op><threshold>'";

const UNIT_COMPOSE_MESSAGE =
  "a unit atom ('neuron/…', 'head/…') does not compose with projection ('~'/'|') or a " +
  "manifold position ('%')";
const MANIFOLD_COMPOSE_MESSAGE =
  "a manifold term does not compose with projection ('~'/'|') or ablation ('!')";

/** Python's `_Parser.parse`: the syntax tree, unresolved. */
export function parseSteeringExpression(text: string): SteeringExpression {
  if (!pythonStrip(text)) throw new SteeringExpressionError("empty steering expression");
  return { terms: new Parser(lex(text)).parse() };
}

/** Every concept atom `(namespace, concept, variant)` (`referenced_selectors`): a projection
 *  gives two, a `%` term none, and a layer clause never appears. */
export function referencedSteeringSelectors(
  text: string,
): Array<Pick<SteeringAtom, "namespace" | "concept" | "variant">> {
  if (!pythonStrip(text)) return [];
  const result: Array<Pick<SteeringAtom, "namespace" | "concept" | "variant">> = [];
  for (const term of parseSteeringExpression(text).terms) {
    if (term.selector.manifoldPosition !== null) continue;
    result.push(stripAtomLocation(term.selector.base));
    if (term.selector.onto !== null) result.push(stripAtomLocation(term.selector.onto));
  }
  return result;
}

function stripAtomLocation(
  atom: SteeringAtom,
): Pick<SteeringAtom, "namespace" | "concept" | "variant"> {
  return { namespace: atom.namespace, concept: atom.concept, variant: atom.variant };
}

const UNIT_REF = /^[Ll]([0-9]+)([NnHh])([0-9]+)$/u;

/** `naming.parse_unit_ref`: `L8n344` / `L20h17` in any letter case, leading zeros dropped. The
 *  letter must match the namespace. Returns the canonical name. */
export function canonicalUnitName(namespace: "neuron" | "head", text: string): string {
  const letter = namespace === "neuron" ? "n" : "h";
  const match = UNIT_REF.exec(text);
  if (match === null || match[2].toLowerCase() !== letter) {
    const example = namespace === "neuron" ? "L8n344" : "L20h17";
    throw new SteeringExpressionError(
      `'${namespace}/${text}' is not a ${namespace} reference; write L<layer>${letter}<index> ` +
        `(e.g. '${namespace}/${example}')`,
    );
  }
  return `L${BigInt(match[1])}${letter}${BigInt(match[3])}`;
}

class Parser {
  private position = 0;

  constructor(private readonly tokens: readonly Token[]) {}

  parse(): SteeringExpressionTerm[] {
    let sign = 1;
    if (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      sign = this.consume().kind === "MINUS" ? -1 : 1;
    }
    const terms = [this.term(sign)];
    while (this.peek().kind === "PLUS" || this.peek().kind === "MINUS") {
      terms.push(this.term(this.consume().kind === "MINUS" ? -1 : 1));
    }
    const trailing = this.peek();
    if (trailing.kind !== "EOF") {
      if (trailing.kind === "IDENT") {
        throw this.error(
          `unexpected identifier ${tokenRepr(trailing)} after a complete term; multi-word ` +
            "concept names use underscores (e.g. 'artificial_intelligence', not 'artificial " +
            "intelligence'), and bipolar pairs join with '.' (e.g. 'human.artificial_intelligence')",
          trailing,
        );
      }
      throw this.error(`unexpected token ${trailing.kind} (${tokenRepr(trailing)})`, trailing);
    }
    return terms;
  }

  private term(sign: number): SteeringExpressionTerm {
    let explicit = false;
    let coefficients = [sign * DEFAULT_STEERING_COEFFICIENT];
    if (this.peek().kind === "NUM") {
      explicit = true;
      coefficients = [sign * Number(this.consume().value)];
      while (this.peek().kind === "COMMA") {
        this.consume();
        coefficients.push(sign * this.signedNumber());
        if (coefficients.length > 2) {
          throw this.error(
            "a manifold coefficient slot takes at most 2 comma-separated values (along, onto); got more",
            this.peek(-1),
          );
        }
      }
      if (this.peek().kind === "STAR") this.consume();
    }
    let ablation = false;
    if (this.peek().kind === "BANG") {
      this.consume();
      ablation = true;
      if (!explicit) coefficients = [sign * DEFAULT_ABLATION_COEFFICIENT];
    }
    const selector = this.selector();
    if (selector.base.namespace === "head" && !ablation) {
      throw this.error(
        `a head has no single write direction; silence it with !head/${selector.base.concept}`,
        selector.base.column,
      );
    }
    let layers: WrittenLayerClause = null;
    if (this.peek().kind === "LAYERS") layers = this.layerClause(selector);
    let trigger: SteeringTrigger | null = null;
    if (this.peek().kind === "AT") {
      this.consume();
      trigger = this.trigger();
      if (this.peek().kind === "LAYERS") {
        throw this.error(
          "the layer clause goes before the '@' trigger (e.g. 'x{4-8}@response'); gate probes " +
            "take no layer clause",
          this.peek(),
        );
      }
    }
    return { coefficients, selector, trigger, ablation, layers };
  }

  private layerClause(selector: SteeringSelector): WrittenLayerClause {
    const token = this.consume();
    for (const atom of [selector.base, selector.onto]) {
      if (atom !== null && atom.namespace !== null && UNIT_NAMESPACES.has(atom.namespace)) {
        throw this.error(
          `'${atom.namespace}/' atoms carry their own layer and take no layer clause`,
          token,
        );
      }
    }
    let parsed: ReturnType<typeof parseLayerSelector>;
    try {
      parsed = parseLayerSelector(String(token.value));
    } catch (error) {
      if (!(error instanceof LayerSelectorError)) throw error;
      throw this.error(`layer clause: ${error.message}`, token.column + 1 + (error.offset ?? 0));
    }
    if (parsed === null) {
      throw this.error("empty layer clause; name layers ('{4-8,20}'), 'workspace' or 'all'", token);
    }
    const next = this.peek();
    if (next.kind === "LAYERS") throw this.error("a term takes one layer clause", next);
    if (next.kind === "TILDE" || next.kind === "ORTHO" || next.kind === "PERCENT") {
      throw this.error(
        "the layer clause follows the whole selector (e.g. 'a~b{4-8}', 'm%x{4-8}')",
        next,
      );
    }
    return Array.isArray(parsed) ? [...parsed].sort((a, b) => a - b) : parsed;
  }

  private trigger(): SteeringTrigger {
    const head = this.expect("IDENT");
    if (head.value === "when") return this.gateClause();
    const phase = this.phase(head);
    if (this.peek().kind !== "AMP") return { phase, gate: null };
    this.consume();
    const gate = this.expect("IDENT");
    if (gate.value !== "when") {
      throw this.error(
        `a compound trigger's second clause must be a probe gate ('&when:<probe><op><threshold>'); got '${gate.value}'`,
        gate,
      );
    }
    return { phase, gate: this.gateClause().gate };
  }

  private phase(token: Token): SteeringPhase {
    const keyword = String(token.value);
    if ((keyword === "first" || keyword === "after") && this.peek().kind === "COLON") {
      this.consume();
      const count = this.expect("NUM");
      const tokens = Number(count.value);
      if (!Number.isSafeInteger(tokens) || tokens < 0) {
        throw this.error(
          `trigger '@${keyword}:' takes a non-negative integer token count, got '${count.value}'`,
          count,
        );
      }
      return { kind: keyword, tokens };
    }
    if (keyword === "first") {
      throw this.error("trigger '@first' requires ':' followed by a token count (e.g. '@first:5')", token);
    }
    const phase = PHASES[keyword];
    if (phase === undefined) {
      throw this.error(
        `unknown trigger '@${keyword}'; ${TRIGGER_FORMS_HELP}.  Note: '@' is for triggers only; ` +
          "HF revisions are not accepted inside steering expressions.",
        token,
      );
    }
    return phase;
  }

  private gateClause(): SteeringTrigger {
    if (this.peek().kind !== "COLON") {
      throw this.error(
        `trigger '@when' requires ':' followed by a probe gate (e.g. '@when:angry.calm>0.4'); got ${this.peek().kind}`,
        this.peek(),
      );
    }
    this.consume();
    return { phase: { kind: "generated_only" }, gate: this.probeGate() };
  }

  private probeGate(): SteeringProbeGate {
    const head = this.expect("IDENT");
    let probe = String(head.value);
    if (this.peek().kind === "SLASH") {
      this.consume();
      const namespace = probe;
      probe = `${probe}/${this.namespacedName(namespace)}`;
      const next = this.peek();
      if (
        UNIT_NAMESPACES.has(namespace) &&
        (next.kind === "COLON" || next.kind === "AT" || next.kind === "TILDE" || next.kind === "LBRACK")
      ) {
        throw this.error(
          `a unit probe ('${probe}') is one scalar and takes no channel suffix ('${next.value}')`,
          next,
        );
      }
    }
    const discriminator = this.peek().kind;
    if (discriminator === "COLON") {
      this.consume();
      const channel = this.expect("IDENT");
      if (channel.value !== "fraction" && channel.value !== "membership") {
        throw this.error(
          `unknown manifold gate channel '${probe}:${channel.value}'; expected '<manifold>:fraction', ` +
            "'<manifold>:membership', '<manifold>@<label>', or '<manifold>~<label>'",
          channel,
        );
      }
      probe = `${probe}:${channel.value}`;
    } else if (discriminator === "AT" || discriminator === "TILDE") {
      this.consume();
      const label = this.expect("IDENT");
      probe = `${probe}${discriminator === "AT" ? "@" : "~"}${label.value}`;
    } else {
      if (discriminator === "DOT") {
        this.consume();
        probe = `${probe}.${String(this.expect("IDENT").value)}`;
      }
      if (this.peek().kind === "LBRACK") {
        this.consume();
        const index = this.expect("NUM");
        const value = Number(index.value);
        if (!Number.isSafeInteger(value) || value < 0) {
          throw this.error(`coordinate index must be a non-negative integer, got '${index.value}'`, index);
        }
        this.expect("RBRACK");
        probe = `${probe}[${value}]`;
      }
    }
    const operator = this.consume();
    if (!["GT", "GTE", "LT", "LTE"].includes(operator.kind)) {
      throw this.error(
        "expected comparison op (>, >=, <, <=) in '@when:' clause, got " +
          `${operator.kind} (${tokenRepr(operator)})`,
        operator,
      );
    }
    return {
      probe,
      operator: ({ GT: ">", GTE: ">=", LT: "<", LTE: "<=" } as const)[
        operator.kind as "GT" | "GTE" | "LT" | "LTE"
      ],
      threshold: this.signedNumber(),
    };
  }

  private selector(): SteeringSelector {
    const base = this.atom();
    const after = this.peek().kind;
    if (
      base.namespace !== null && UNIT_NAMESPACES.has(base.namespace) &&
      (after === "PERCENT" || after === "TILDE" || after === "ORTHO")
    ) {
      throw this.error(UNIT_COMPOSE_MESSAGE, this.peek());
    }
    if (after === "PERCENT") {
      this.consume();
      let manifoldPosition: readonly number[] | string;
      if (this.peek().kind === "IDENT") {
        manifoldPosition = String(this.consume().value);
      } else {
        const coordinates = [this.signedNumber()];
        while (this.peek().kind === "COMMA") {
          this.consume();
          coordinates.push(this.signedNumber());
        }
        manifoldPosition = coordinates;
      }
      if (["TILDE", "ORTHO", "PERCENT"].includes(this.peek().kind)) {
        throw this.error(MANIFOLD_COMPOSE_MESSAGE, this.peek());
      }
      return { base, projection: null, onto: null, manifoldPosition };
    }
    if (after === "TILDE" || after === "ORTHO") {
      const projection = this.consume().kind === "TILDE" ? "~" : "|";
      const onto = this.atom();
      if (onto.namespace !== null && UNIT_NAMESPACES.has(onto.namespace)) {
        throw this.error(UNIT_COMPOSE_MESSAGE, onto.column);
      }
      if (this.peek().kind === "TILDE" || this.peek().kind === "ORTHO") {
        throw this.error("chained projection is not allowed; use one '~' or '|' per term", this.peek());
      }
      return { base, projection, onto, manifoldPosition: null };
    }
    return { base, projection: null, onto: null, manifoldPosition: null };
  }

  private atom(): SteeringAtom {
    const first = this.expect("IDENT");
    let namespace: string | null = null;
    let concept = String(first.value);
    if (this.peek().kind === "SLASH") {
      this.consume();
      namespace = concept;
      concept = this.namespacedName(namespace);
    }
    if (this.peek().kind === "DOT") {
      this.consume();
      concept = `${concept}.${String(this.expect("IDENT").value)}`;
    }
    let variant = "raw";
    if (this.peek().kind === "COLON") {
      this.consume();
      const suffix = this.expect("IDENT");
      variant = String(suffix.value);
      if (namespace !== null && RESERVED_NAMESPACES.has(namespace)) {
        throw this.error(
          `'${namespace}/' atoms name instrument units and take no ':${variant}' variant`,
          suffix,
        );
      }
    }
    return { namespace, concept, variant, column: first.column };
  }

  /** The name after `<namespace>/`: a unit under `neuron/` and `head/` (canonical), a feature id
   *  under `sae/`, an exact vocab id `#<int>` under the lens namespaces, else an identifier. */
  private namespacedName(namespace: string): string {
    if (namespace === "neuron" || namespace === "head") return this.unitName(namespace);
    if (namespace === "sae" && this.peek().kind === "NUM") {
      return String(this.nonNegativeInteger("SAE feature id"));
    }
    if (TOKEN_ID_NAMESPACES.has(namespace) && this.peek().kind === "HASH") {
      const hash = this.consume();
      if (this.peek().column !== hash.column + 1) {
        throw this.error("token id must follow '#' with no space", this.peek());
      }
      return `#${this.nonNegativeInteger("token id")}`;
    }
    return String(this.expect("IDENT").value);
  }

  private unitName(namespace: "neuron" | "head"): string {
    const token = this.peek();
    if (token.kind !== "IDENT") {
      const example = namespace === "neuron" ? "L8n344" : "L20h17";
      throw this.error(`'${namespace}/' takes a unit, e.g. '${namespace}/${example}'`, token);
    }
    this.consume();
    let unit: string;
    try {
      unit = canonicalUnitName(namespace, String(token.value));
    } catch (error) {
      throw this.error(error instanceof Error ? error.message.replace(/ \(col \d+\)$/u, "") : String(error), token);
    }
    if (this.peek().kind === "DOT") {
      throw this.error(`'${namespace}/${unit}' is a whole unit and takes no '.'`, this.peek());
    }
    return unit;
  }

  private nonNegativeInteger(what: string): number {
    const token = this.expect("NUM");
    const value = Number(token.value);
    if (!Number.isSafeInteger(value) || value < 0) {
      throw this.error(`${what} must be a non-negative integer`, token);
    }
    return value;
  }

  private signedNumber(): number {
    let sign = 1;
    if (this.peek().kind === "MINUS") {
      this.consume();
      sign = -1;
    } else if (this.peek().kind === "PLUS") {
      this.consume();
    }
    return sign * Number(this.expect("NUM").value);
  }

  private peek(offset = 0): Token {
    return this.tokens[this.position + offset];
  }

  private consume(): Token {
    return this.tokens[this.position++];
  }

  private expect(kind: TokenKind): Token {
    const token = this.peek();
    if (token.kind !== kind) {
      throw this.error(`expected ${kind}, got ${token.kind} (${tokenRepr(token)})`, token);
    }
    return this.consume();
  }

  private error(message: string, location: Token | number): SteeringExpressionError {
    return new SteeringExpressionError(
      message,
      typeof location === "number" ? location : location.column,
    );
  }
}

// ------------------------------------------------------------------ the IR ---

/** One entry of `Steering.alphas`, in written order. `trigger` is `null` for `Trigger.BOTH`. */
export type SteeringIrTerm =
  | { kind: "vector"; key: string; target: string; coeff: number; layers: LayerScope; trigger: SteeringTrigger | null }
  | { kind: "ablation"; key: string; target: string; coeff: number; layers: LayerScope; trigger: SteeringTrigger | null }
  | { kind: "unitAblation"; key: string; unit: string; coeff: number; trigger: SteeringTrigger | null }
  | {
      kind: "projection";
      key: string;
      base: string;
      operator: SteeringProjectionOperator;
      onto: string;
      coeff: number;
      layers: LayerScope;
      trigger: SteeringTrigger | null;
    }
  | {
      kind: "manifold";
      key: string;
      manifold: string;
      position: readonly number[] | string;
      along: number;
      onto: number;
      layers: LayerScope;
      trigger: SteeringTrigger | null;
    };

export interface SteeringIr {
  terms: readonly SteeringIrTerm[];
}

/** An installed manifold, as the bare-atom ladder sees it (`store.selectors` index). */
export interface SteeringManifoldEntry {
  namespace: string;
  name: string;
  fitMode: string;
  labels: readonly string[];
}

export interface SteeringResolution {
  /** Installed manifolds: node labels (tier 1) and 2-node `pca` names (tier 2). */
  manifolds?: readonly SteeringManifoldEntry[];
  /** Session-local profiles whose exact key wins over the manifold tiers. */
  profileNames?: Iterable<string>;
  /** The expression-level namespace scope (`parse_expr(namespace=)`). */
  namespace?: string | null;
}

/** `parse_expr`: parse, then fold into the IR. Bare atoms resolve against `resolution`. */
export function parseSteering(text: string, resolution: SteeringResolution = {}): SteeringIr {
  if (!text || !pythonStrip(text)) throw new SteeringExpressionError("empty steering expression");
  return foldSteering(parseSteeringExpression(text), resolution);
}

const VARIANT_SUFFIX = /^(raw|sae(?:-[a-z0-9._-]+)?|role(?:-[a-z0-9._-]+)?|from(?:-[a-z0-9._-]+)?)$/u;

function slug(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/gu, "_").replace(/^_+|_+$/gu, "");
}

/** `naming.canonical_concept_name`: lower-case and slug; a pre-joined `a.b` slugs each side. */
export function canonicalConceptName(concept: string): string {
  const separator = concept.indexOf(".");
  if (separator < 0) return slug(concept);
  return `${slug(concept.slice(0, separator))}.${slug(concept.slice(separator + 1))}`;
}

/** `selectors.canonicalize_atom`: peel a `:variant` suffix and canonicalize the name. */
function canonicalizeAtom(raw: string): [string, string] {
  let variant = "raw";
  let name = raw;
  const colon = raw.lastIndexOf(":");
  if (colon >= 0) {
    const suffix = raw.slice(colon + 1);
    if (!VARIANT_SUFFIX.test(suffix)) {
      throw new SteeringSelectorError(`unknown variant '${suffix}' in '${raw}'`);
    }
    variant = suffix;
    name = raw.slice(0, colon);
  }
  return [canonicalConceptName(name), variant];
}

function withVariant(canonical: string, variant: string): string {
  return variant === "raw" ? canonical : `${canonical}:${variant}`;
}

/** A pole atom's key (`_resolve_atom`): reserved atoms verbatim, else the canonical slug with
 *  the typed namespace and a non-raw variant. */
function resolveAtomKey(atom: SteeringAtom): string {
  if (atom.namespace !== null && RESERVED_NAMESPACES.has(atom.namespace)) {
    return `${atom.namespace}/${atom.concept}`;
  }
  const [canonical, variant] = canonicalizeAtom(
    atom.variant === "raw" ? atom.concept : `${atom.concept}:${atom.variant}`,
  );
  return withVariant(atom.namespace === null ? canonical : `${atom.namespace}/${canonical}`, variant);
}

/** A manifold atom's key (`_resolve_manifold_atom`): the name verbatim, no ladder. */
function manifoldAtomKey(atom: SteeringAtom): string {
  return withVariant(atom.namespace === null ? atom.concept : `${atom.namespace}/${atom.concept}`, atom.variant);
}

function formatPosition(position: readonly number[] | string): string {
  return typeof position === "string" ? position : position.map((value) => formatG(value, 6)).join(",");
}

function exactPosition(position: readonly number[] | string): string {
  return typeof position === "string" ? position : position.map(pythonFloatRepr).join(",");
}

/** Python's `repr` of a float: the shortest round-trip digits, fixed between 1e-4 and 1e16. */
function pythonFloatRepr(value: number): string {
  if (Number.isNaN(value)) return "nan";
  if (!Number.isFinite(value)) return value > 0 ? "inf" : "-inf";
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";
  const [mantissa, exponentText] = Math.abs(value).toExponential().split("e");
  const exponent = Number(exponentText);
  const digits = mantissa.replace(".", "");
  const sign = value < 0 ? "-" : "";
  if (exponent < -4 || exponent >= 16) {
    const scaled = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${scaled}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  if (exponent >= digits.length - 1) return `${sign}${digits}${"0".repeat(exponent - digits.length + 1)}.0`;
  if (exponent >= 0) return `${sign}${digits.slice(0, exponent + 1)}.${digits.slice(exponent + 1)}`;
  return `${sign}0.${"0".repeat(-exponent - 1)}${digits}`;
}

const NON_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/** Python's `repr` of a string: single quotes unless only double quotes avoid escaping. */
function pythonStringRepr(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let output = quote;
  for (const character of text) {
    const code = character.codePointAt(0)!;
    if (character === quote || character === "\\") output += `\\${character}`;
    else if (character === "\n") output += "\\n";
    else if (character === "\r") output += "\\r";
    else if (character === "\t") output += "\\t";
    else if (character !== " " && NON_PRINTABLE.test(character)) {
      const hex = code.toString(16);
      output += code < 0x100
        ? `\\x${hex.padStart(2, "0")}`
        : code < 0x10000 ? `\\u${hex.padStart(4, "0")}` : `\\U${hex.padStart(8, "0")}`;
    } else output += character;
  }
  return output + quote;
}

/** A token's value as Python's `{t.value!r}` writes it. */
function tokenRepr(token: Token): string {
  return typeof token.value === "number" ? pythonFloatRepr(token.value) : pythonStringRepr(token.value);
}

function scopeOf(term: SteeringExpressionTerm): LayerScope {
  return term.layers === null || term.layers === "all" ? null : term.layers;
}

/** `Trigger.BOTH` is `null`. */
export function normalizeTrigger(trigger: SteeringTrigger | null): SteeringTrigger | null {
  return trigger === null || (trigger.phase.kind === "both" && trigger.gate === null) ? null : trigger;
}

/** A trigger as Python's `Trigger` fields: which windows it steers and its counted bounds. */
export function steeringTriggerFields(trigger: SteeringTrigger | null): {
  prompt: boolean;
  generated: boolean;
  thinking: boolean;
  response: boolean;
  firstN: number | null;
  afterN: number | null;
} {
  const phase = trigger?.phase ?? { kind: "both" as const };
  return {
    prompt: phase.kind === "both" || phase.kind === "prompt_only",
    generated: phase.kind !== "prompt_only",
    thinking: phase.kind !== "after_thinking",
    response: phase.kind !== "thinking_only",
    firstN: phase.kind === "first" ? phase.tokens : null,
    afterN: phase.kind === "after" ? phase.tokens : null,
  };
}

/** Equal triggers (dataclass equality, so `-0.0 == 0.0`). */
export function steeringTriggersEqual(a: SteeringTrigger | null, b: SteeringTrigger | null): boolean {
  const left = normalizeTrigger(a);
  const right = normalizeTrigger(b);
  if (left === null || right === null) return left === right;
  if (left.phase.kind !== right.phase.kind) return false;
  if ("tokens" in left.phase && left.phase.tokens !== (right.phase as { tokens: number }).tokens) return false;
  if (left.gate === null || right.gate === null) return left.gate === right.gate;
  return left.gate.probe === right.gate.probe &&
    left.gate.operator === right.gate.operator &&
    left.gate.threshold === right.gate.threshold;
}

class Folder {
  readonly terms: SteeringIrTerm[] = [];
  private readonly index = new Map<string, number>();

  get(key: string): SteeringIrTerm | undefined {
    const at = this.index.get(key);
    return at === undefined ? undefined : this.terms[at];
  }

  put(term: SteeringIrTerm): void {
    const at = this.index.get(term.key);
    if (at === undefined) {
      this.index.set(term.key, this.terms.length);
      this.terms.push(term);
    } else {
      this.terms[at] = term;
    }
  }
}

function conflicting(what: string, key: string): SteeringExpressionError {
  return new SteeringExpressionError(
    `${what} '${key}' appears with conflicting triggers; merge triggers explicitly or split ` +
      "into separate Steering entries",
  );
}

function mergePlain(
  folder: Folder, key: string, target: string, layers: LayerScope, coeff: number,
  trigger: SteeringTrigger | null,
): void {
  const existing = folder.get(key);
  if (existing === undefined) {
    folder.put({ kind: "vector", key, target, coeff, layers, trigger });
    return;
  }
  if (existing.kind === "projection") {
    throw new SteeringExpressionError(
      `concept '${key}' appears both as a plain term and as a projection target; use distinct references`,
    );
  }
  if (existing.kind !== "vector" || !steeringTriggersEqual(existing.trigger, trigger)) {
    throw conflicting("concept", key);
  }
  folder.put({ ...existing, coeff: existing.coeff + coeff });
}

function mergeProjected(
  folder: Folder, key: string, operator: SteeringProjectionOperator, base: string, onto: string,
  coeff: number, trigger: SteeringTrigger | null, layers: LayerScope,
): void {
  const existing = folder.get(key);
  if (existing === undefined) {
    folder.put({ kind: "projection", key, base, operator, onto, coeff, layers, trigger });
    return;
  }
  if (existing.kind !== "projection") {
    throw new SteeringExpressionError(`projection '${key}' conflicts with a plain entry of the same name`);
  }
  if (!steeringTriggersEqual(existing.trigger, trigger)) {
    throw new SteeringExpressionError(`projection '${key}' appears with conflicting triggers`);
  }
  folder.put({ ...existing, coeff: existing.coeff + coeff });
}

function mergeAblation(
  folder: Folder, target: string, coeff: number, trigger: SteeringTrigger | null, layers: LayerScope,
): void {
  const key = `!${target}${formatLayerClause(layers)}`;
  const existing = folder.get(key);
  if (existing === undefined) {
    folder.put({ kind: "ablation", key, target, coeff, layers, trigger });
    return;
  }
  if (existing.kind !== "ablation" || !steeringTriggersEqual(existing.trigger, trigger)) {
    throw conflicting("ablation", key);
  }
  folder.put({ ...existing, coeff: existing.coeff + coeff });
}

function mergeUnitAblation(
  folder: Folder, unit: string, coeff: number, trigger: SteeringTrigger | null,
): void {
  const key = `!${unit}`;
  const existing = folder.get(key);
  if (existing === undefined) {
    folder.put({ kind: "unitAblation", key, unit, coeff, trigger });
    return;
  }
  if (existing.kind !== "unitAblation" || !steeringTriggersEqual(existing.trigger, trigger)) {
    throw conflicting("unit ablation", key);
  }
  folder.put({ ...existing, coeff: existing.coeff + coeff });
}

function mergeManifold(
  folder: Folder, manifold: string, coefficients: readonly number[],
  position: readonly number[] | string, trigger: SteeringTrigger | null, layers: LayerScope,
): void {
  const along = coefficients[0];
  const onto = coefficients.length === 1 ? 0 : coefficients[1];
  const key = `${manifold}%${formatPosition(position)}${formatLayerClause(layers)}`;
  const existing = folder.get(key);
  if (existing === undefined) {
    folder.put({ kind: "manifold", key, manifold, position, along, onto, layers, trigger });
    return;
  }
  if (existing.kind !== "manifold" || !steeringTriggersEqual(existing.trigger, trigger)) {
    throw conflicting("manifold", key);
  }
  const same = typeof existing.position === "string" || typeof position === "string"
    ? existing.position === position
    : existing.position.length === position.length &&
      existing.position.every((value, index) => value === position[index]);
  if (!same) {
    // The key spells coordinates in %g (six significant digits), so two positions can share it;
    // summing them would move one term silently onto the other's point.
    throw new SteeringExpressionError(
      `manifold '${key}' names two positions that agree to six significant digits ` +
        `(${exactPosition(existing.position)} and ${exactPosition(position)}); round them to one ` +
        "point or move them further apart",
    );
  }
  folder.put({ ...existing, along: existing.along + along, onto: existing.onto + onto });
}

/** The bare-atom ladder (`selectors.resolve_bare_atom`): a node label, then a 2-node `pca`
 *  manifold name, else `null` (the pole tier: a plain vector). */
function resolveBareAtom(
  atom: SteeringAtom,
  manifolds: readonly SteeringManifoldEntry[],
  namespace: string | null,
): { manifold: string; position: string } | null {
  if (atom.variant === "raw" && atom.namespace === null && !atom.concept.includes(".")) {
    const hits = manifolds.filter((entry) =>
      entry.labels.includes(atom.concept) && (namespace === null || entry.namespace === namespace)
    );
    if (hits.length > 1) {
      const qualified = hits
        .map((entry) => `${entry.namespace}/${entry.name} (label '${atom.concept}')`)
        .join(", ");
      throw new AmbiguousSteeringSelectorError(
        `ambiguous manifold label '${atom.concept}': matches ${qualified}. Specify the manifold ` +
          `explicitly as <ns>/<name>%${atom.concept}.`,
      );
    }
    if (hits.length === 1) {
      return { manifold: `${hits[0].namespace}/${hits[0].name}`, position: atom.concept };
    }
  }
  if (atom.variant === "raw") {
    const scope = atom.namespace ?? namespace;
    const hits = manifolds.filter((entry) =>
      entry.fitMode === "pca" && entry.labels.length === 2 && entry.name === atom.concept &&
      (scope === null || entry.namespace === scope)
    );
    if (hits.length > 1) {
      throw new AmbiguousSteeringSelectorError(
        `ambiguous manifold name '${atom.concept}': matches ` +
          `${hits.map((entry) => `${entry.namespace}/${entry.name}`).join(", ")}. Specify the ` +
          `namespace as <ns>/${atom.concept}.`,
      );
    }
    if (hits.length === 1) {
      return { manifold: `${hits[0].namespace}/${hits[0].name}`, position: hits[0].labels[0] };
    }
  }
  return null;
}

const MULTI_COEFFICIENT_MESSAGE =
  "comma-separated coefficients are only valid for `manifold % position` terms";

/** `grammar._fold`: the syntax tree into the IR, merging repeated keys. */
export function foldSteering(
  parsed: SteeringExpression,
  resolution: SteeringResolution = {},
): SteeringIr {
  const folder = new Folder();
  const manifolds = resolution.manifolds ?? [];
  const profiles = new Set(resolution.profileNames ?? []);
  const namespace = resolution.namespace ?? null;
  for (const term of parsed.terms) {
    const selector = term.selector;
    const base = selector.base;
    const scope = scopeOf(term);
    const clause = formatLayerClause(scope);
    const trigger = normalizeTrigger(term.trigger);
    if (selector.manifoldPosition !== null) {
      if (term.ablation) throw new SteeringExpressionError(MANIFOLD_COMPOSE_MESSAGE);
      mergeManifold(
        folder, manifoldAtomKey(base), term.coefficients, selector.manifoldPosition, trigger, scope,
      );
      continue;
    }
    if (
      selector.projection === null && !term.ablation &&
      !(base.namespace !== null && RESERVED_NAMESPACES.has(base.namespace))
    ) {
      const profileKey = resolveAtomKey(base);
      if (profiles.has(profileKey)) {
        if (term.coefficients.length > 1) throw new SteeringExpressionError(MULTI_COEFFICIENT_MESSAGE);
        mergePlain(folder, profileKey + clause, profileKey, scope, term.coefficients[0], trigger);
        continue;
      }
      const hit = resolveBareAtom(base, manifolds, namespace);
      if (hit !== null) {
        mergeManifold(folder, hit.manifold, term.coefficients, hit.position, trigger, scope);
        continue;
      }
    }
    if (term.coefficients.length > 1) throw new SteeringExpressionError(MULTI_COEFFICIENT_MESSAGE);
    const baseKey = resolveAtomKey(base);
    const coeff = term.coefficients[0];
    if (term.ablation) {
      if (selector.projection !== null) {
        throw new SteeringExpressionError(
          "ablation does not compose with projection operators; ablate the base concept and " +
            "project separately",
        );
      }
      if (base.namespace !== null && UNIT_NAMESPACES.has(base.namespace)) {
        mergeUnitAblation(folder, baseKey, coeff, trigger);
      } else {
        mergeAblation(folder, baseKey, coeff, trigger, scope);
      }
      continue;
    }
    if (selector.projection === null) {
      mergePlain(folder, baseKey + clause, baseKey, scope, coeff, trigger);
      continue;
    }
    const ontoKey = resolveAtomKey(selector.onto!);
    mergeProjected(
      folder, `${baseKey}${selector.projection}${ontoKey}${clause}`, selector.projection, baseKey,
      ontoKey, coeff, trigger, scope,
    );
  }
  return { terms: folder.terms };
}

// ------------------------------------------------------------------ format ---

/** A number as `format_expr` writes it: Python's `%g`. */
export function formatSteeringNumber(value: number): string {
  return formatG(value, 6);
}

const PHASE_NAMES: Partial<
  Record<SteeringPhase["kind"], "both" | "before" | "response" | "after" | "thinking">
> = {
  both: "both",
  after_thinking: "after",
  prompt_only: "before",
  thinking_only: "thinking",
  generated_only: "response",
};

function phaseText(phase: SteeringPhase): string {
  if (phase.kind === "first" || phase.kind === "after") return `${phase.kind}:${phase.tokens}`;
  return PHASE_NAMES[phase.kind]!;
}

/** A trigger's `@` clause without the `@` (`_trigger_name`); `null` for `Trigger.BOTH`. A bare
 *  gate's own window (`response`) renders without a phase. */
export function formatSteeringTrigger(trigger: SteeringTrigger | null): string | null {
  const trig = normalizeTrigger(trigger);
  if (trig === null) return null;
  if (trig.gate === null) return phaseText(trig.phase);
  const gate = `when:${trig.gate.probe}${trig.gate.operator}${formatSteeringNumber(trig.gate.threshold)}`;
  return trig.phase.kind === "generated_only" ? gate : `${phaseText(trig.phase)}&${gate}`;
}

function withTrigger(body: string, trigger: SteeringTrigger | null): string {
  const text = formatSteeringTrigger(trigger);
  return text === null ? body : `${body}@${text}`;
}

function formatBang(target: string, coeff: number, trigger: SteeringTrigger | null): string {
  const body = coeff === 1
    ? `!${target}`
    : coeff === -1 ? `-!${target}` : `${formatSteeringNumber(coeff)} !${target}`;
  return withTrigger(body, trigger);
}

/** One term's text, its leading sign included (`TermPart.text`). */
export function formatSteeringTerm(term: SteeringIrTerm): string {
  switch (term.kind) {
    case "vector":
      return withTrigger(`${formatSteeringNumber(term.coeff)} ${term.key}`, term.trigger);
    case "ablation":
      return formatBang(`${term.target}${formatLayerClause(term.layers)}`, term.coeff, term.trigger);
    case "unitAblation":
      return formatBang(term.unit, term.coeff, term.trigger);
    case "projection":
      return withTrigger(
        `${formatSteeringNumber(term.coeff)} ${term.base}${term.operator}${term.onto}` +
          formatLayerClause(term.layers),
        term.trigger,
      );
    case "manifold":
      return withTrigger(
        `${formatAlongOnto(term.along, term.onto)} ${term.manifold}%${formatPosition(term.position)}` +
          formatLayerClause(term.layers),
        term.trigger,
      );
  }
}

/** A `%` term's `along[,onto]`: `onto` renders relative to `along`'s sign bit, which the parser
 *  applies to the whole run. */
function formatAlongOnto(along: number, onto: number): string {
  const negative = along < 0 || Object.is(along, -0);
  return onto === 0
    ? formatSteeringNumber(along)
    : `${formatSteeringNumber(along)},${formatSteeringNumber(negative ? -onto : onto)}`;
}

/** The loom's `inverted` regen (`Recipe.invert_steering`): `parse_expr` against the installed
 *  manifolds (`resolution.manifolds`; drowse passes no profile names), so bare labels resolve to
 *  their `%` terms and repeats fold, then every term's coefficient negated (a `%` term's `along`,
 *  a zero staying `+0`; its `onto` carried) and the whole written back by `format_expr`. A
 *  refusal is drowse's too (an ambiguous label, a two-value run on a term that is not `%`). */
export function invertSteeringExpression(
  text: string,
  resolution: Pick<SteeringResolution, "manifolds"> = {},
): string {
  if (!pythonStrip(text)) return "";
  const ir = parseSteering(text, { manifolds: resolution.manifolds ?? [] });
  return formatSteering({
    terms: ir.terms.map((term) =>
      term.kind === "manifold" ? { ...term, along: -term.along || 0 } : { ...term, coeff: -term.coeff }
    ),
  });
}

/** `format_expr`: the canonical text, its terms joined by ` + ` or ` - `. */
export function formatSteering(ir: SteeringIr): string {
  return joinSteeringTerms(ir.terms.map(formatSteeringTerm));
}

function joinSteeringTerms(texts: readonly string[]): string {
  let output = "";
  texts.forEach((text, index) => {
    if (index === 0) output = text;
    else if (text.startsWith("-")) output += ` - ${text.slice(1).trimStart()}`;
    else output += ` + ${text}`;
  });
  return output;
}

/** `term_parts` on the wire (`session_routes._part_json`): each term with its fields. */
export function steeringParts(ir: SteeringIr): SteeringPartJSON[] {
  return ir.terms.map((term): SteeringPartJSON => {
    const trigger = term.trigger;
    const window: SteeringPhase = trigger?.phase ?? { kind: "both" };
    const shared = {
      text: formatSteeringTerm(term),
      trigger: formatSteeringTrigger(trigger),
      phase: PHASE_NAMES[window.kind] ?? null,
      first_n: window.kind === "first" ? window.tokens : null,
      after_n: window.kind === "after" ? window.tokens : null,
      gate: trigger?.gate
        ? { probe: trigger.gate.probe, op: trigger.gate.operator, threshold: trigger.gate.threshold }
        : null,
    };
    switch (term.kind) {
      case "vector":
      case "ablation":
        return {
          ...shared, op: term.kind === "vector" ? "push" : "ablate", target: term.target,
          coeff: term.coeff, onto: 0, position: null, layers: layersJson(term.layers),
          projection: null,
        };
      case "unitAblation":
        return {
          ...shared, op: "ablate", target: term.unit, coeff: term.coeff, onto: 0, position: null,
          layers: null, projection: null,
        };
      case "projection":
        return {
          ...shared, op: "project", target: `${term.base}${term.operator}${term.onto}`,
          coeff: term.coeff, onto: 0, position: null, layers: layersJson(term.layers),
          projection: { base: term.base, operator: term.operator, onto: term.onto },
        };
      case "manifold":
        return {
          ...shared, op: "position", target: term.manifold, coeff: term.along, onto: term.onto,
          position: typeof term.position === "string" ? term.position : [...term.position],
          layers: layersJson(term.layers), projection: null,
        };
    }
  });
}

function layersJson(layers: LayerScope): "workspace" | number[] | null {
  return layers === null || layers === "workspace" ? layers : [...layers];
}
