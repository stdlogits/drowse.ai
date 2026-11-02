import type {
  SteeringAtom,
  SteeringExpression,
  SteeringExpressionTerm,
} from "./steeringExpression";
import { parseSteeringExpression, UNIT_NAMESPACES } from "./steeringExpression";

/** What a steering expression asks of the loaded model library's hooks.
 *
 *  A `{…}` layer clause asks nothing new: every program is laid out per layer, so a clause only
 *  chooses which layers a term's entries fill. `units` covers `neuron/` and `head/` atoms (a
 *  neuron push, a unit ablation `!neuron/…`/`!head/…`); a unit ablation is a mean-ablation
 *  inside the block, not the residual `ablation`. `slens` covers `slens/` atoms. Both wait on
 *  hook ABI v5 (Half B): until then no library offers them, and the refusal is the compiler's
 *  (`instrumentAtomUnavailable`).
 *
 *  A gate asks only for `probe_gate`. Its probe is the name of an attached probe, as in drowse
 *  (the scalar key an attached instrument emits), and any name can alias any family, so the
 *  expression alone cannot say what the gate reads. The compiler resolves it against the
 *  attached probes: it installs lens and SAE gates by the attached probe's family and refuses an
 *  unattached unit or S-lens gate with `instrumentAtomUnavailable`. */
export type DrowseHookFeature =
  | "rank_one"
  | "multi_term"
  | "projection"
  | "ablation"
  | "curved_manifold"
  | "phase_trigger"
  | "probe_gate"
  | "sae"
  | "jlens"
  | "slens"
  | "units";

export type DrowseHookCapabilities = Readonly<Record<DrowseHookFeature, boolean>>;

export const AFFINE_FEASIBILITY_CAPABILITIES: DrowseHookCapabilities = Object.freeze({
  rank_one: true,
  multi_term: true,
  projection: true,
  ablation: true,
  curved_manifold: false,
  phase_trigger: false,
  probe_gate: false,
  sae: false,
  jlens: false,
  slens: false,
  units: false,
});

/** The instrument namespaces whose atoms and gates the browser cannot run yet, in the order a
 *  refusal names them (the first one written). */
const INSTRUMENT_FEATURES: ReadonlyMap<string, DrowseHookFeature> = new Map([
  ["neuron", "units"],
  ["head", "units"],
  ["slens", "slens"],
]);

/** The compiler's refusal for a `neuron/`, `head/` or `slens/` atom or gate the runtime cannot
 *  run (browserCorePack resolves them to this before any hook check). */
export function instrumentAtomUnavailable(namespace: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(
    new Error(`${namespace}/ instrument atoms are not available in the browser runtime`),
    { code: "CORE_PACK_EXPRESSION_UNAVAILABLE", status: 409, recoverable: true },
  );
}

export function requiredHookFeatures(
  expression: string | SteeringExpression,
): ReadonlySet<DrowseHookFeature> {
  const parsed = typeof expression === "string"
    ? parseSteeringExpression(expression)
    : expression;
  const features = new Set<DrowseHookFeature>(["rank_one"]);
  if (parsed.terms.length > 1) features.add("multi_term");
  for (const term of parsed.terms) collectTermFeatures(term, features);
  return features;
}

export function unsupportedHookFeatures(
  required: ReadonlySet<DrowseHookFeature>,
  capabilities: DrowseHookCapabilities,
): DrowseHookFeature[] {
  return [...required].filter((feature) => !capabilities[feature]).sort();
}

export function assertHookFeaturesSupported(
  expression: string | SteeringExpression,
  capabilities: DrowseHookCapabilities,
): void {
  const parsed = typeof expression === "string"
    ? parseSteeringExpression(expression)
    : expression;
  const missing = unsupportedHookFeatures(requiredHookFeatures(parsed), capabilities);
  if (missing.length === 0) return;
  // A unit or S-lens atom the library cannot run gets the compiler's words, so the person reads
  // one refusal whichever check meets it first.
  for (const namespace of instrumentNamespaces(parsed)) {
    const feature = INSTRUMENT_FEATURES.get(namespace);
    if (feature !== undefined && !capabilities[feature]) {
      throw Object.assign(instrumentAtomUnavailable(namespace), { missingFeatures: missing });
    }
  }
  throw Object.assign(
    new Error(`The browser model library does not support: ${missing.join(", ")}`),
    {
      code: "DROWSE_HOOK_FEATURE_UNAVAILABLE",
      missingFeatures: missing,
    },
  );
}

function collectTermFeatures(
  term: SteeringExpressionTerm,
  features: Set<DrowseHookFeature>,
): void {
  const unit = isUnitAtom(term.selector.base);
  if (term.selector.projection !== null) features.add("projection");
  // `!neuron/…` mean-ablates the unit inside the block (units), not a residual direction.
  if (term.ablation && !unit) features.add("ablation");
  if (term.selector.manifoldPosition !== null) features.add("curved_manifold");
  if (term.trigger !== null) {
    if (term.trigger.phase.kind !== "both") features.add("phase_trigger");
    if (term.trigger.gate !== null) features.add("probe_gate");
  }
  for (const atom of [term.selector.base, term.selector.onto]) {
    if (atom === null) continue;
    if (atom.variant === "sae" || atom.variant.startsWith("sae-")) features.add("sae");
    const feature = namespaceFeature(atom.namespace);
    if (feature !== null) features.add(feature);
  }
}

function namespaceFeature(namespace: string | null): DrowseHookFeature | null {
  if (namespace === null) return null;
  if (namespace === "sae" || namespace === "jlens") return namespace;
  return INSTRUMENT_FEATURES.get(namespace) ?? null;
}

function isUnitAtom(atom: SteeringAtom): boolean {
  return atom.namespace !== null && UNIT_NAMESPACES.has(atom.namespace);
}

/** Every atom namespace the expression names, in the order the compiler resolves them: each
 *  term's base, then its onto. */
function instrumentNamespaces(parsed: SteeringExpression): string[] {
  const namespaces: string[] = [];
  for (const term of parsed.terms) {
    for (const atom of [term.selector.base, term.selector.onto]) {
      if (atom?.namespace) namespaces.push(atom.namespace);
    }
  }
  return namespaces;
}
