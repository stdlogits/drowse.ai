import {
  isRuntimeServiceMethod,
  RUNTIME_SERVICE_METHODS,
  type RuntimeClient,
  type RuntimeCapabilityOperation,
  type RuntimeServiceMap,
  type RuntimeServiceName,
  type RuntimeServiceRequest,
} from "../../../../drowse/webui/src/lib/runtime/contracts";
import type { RuntimeFailure } from "../../../../drowse/webui/src/lib/runtime/contracts";
import { runtimeFailure } from "../../lib/runtime/state";

export type RuntimeServiceRoute =
  | "artifact"
  | "hybrid_manifold_read"
  | "fitting"
  | "model"
  | "unavailable";

export interface RuntimeServiceOperationPolicy {
  lifecycle: "ordinary" | "fitting";
  requiredCapabilities: readonly RuntimeCapabilityOperation[];
}

export const RUNTIME_SERVICE_ROUTES = Object.freeze({
  sessions: Object.freeze({
    operationStatus: "model",
    list: "model",
    get: "model",
    patch: "model",
    validateSteering: "model",
  }),
  profiles: Object.freeze({
    list: "model",
    get: "model",
    correlation: "model",
    pairwise: "model",
    extract: "fitting",
  }),
  probes: Object.freeze({
    list: "model",
    attach: "model",
    detach: "model",
    geometry: "model",
  }),
  manifolds: Object.freeze({
    list: "hybrid_manifold_read",
    get: "hybrid_manifold_read",
    create: "artifact",
    createDiscover: "artifact",
    createFromTemplate: "artifact",
    delete: "artifact",
    search: "artifact",
    install: "artifact",
    merge: "artifact",
    fit: "fitting",
    generate: "fitting",
    drowseArchiveList: "artifact",
    drowseArchiveInstall: "artifact",
    drowseArchiveExport: "artifact",
    drowseArchiveDelete: "artifact",
  }),
  templates: Object.freeze({
    list: "artifact",
    get: "artifact",
    create: "artifact",
    delete: "artifact",
    score: "fitting",
  }),
  tree: Object.freeze({
    replayCapabilities: "model",
    get: "model",
    reset: "model",
    restore: "model",
    active: "model",
    navigate: "model",
    edit: "model",
    branch: "model",
    delete: "model",
    star: "model",
    note: "model",
    edgeLabel: "model",
    filter: "model",
    diff: "model",
    transcriptExport: "model",
    transcriptLoad: "model",
    jointLogprobs: "model",
    cast: "model",
    castPut: "model",
    castDelete: "model",
  }),
  instruments: Object.freeze({
    setLive: "model",
    sources: "model",
    setLensSource: "model",
    activateInstalledPack: "model",
    startPreparation: "model",
    preparationStatus: "model",
    cancelPreparation: "model",
    tokenReadout: "model",
    validateLensToken: "model",
    validateSaeFeature: "model",
    saeFeaturesMetadata: "model",
    tokenAnatomy: "model",
    unitActivity: "model",
    unitRead: "model",
    unitTop: "model",
    unitFind: "model",
    unitCatalog: "model",
    unitLabels: "model",
  }),
}) satisfies {
  [Service in RuntimeServiceName]: Readonly<
    Record<
      Extract<keyof RuntimeServiceMap[Service], string>,
      RuntimeServiceRoute
    >
  >;
};

const ORDINARY: RuntimeServiceOperationPolicy = Object.freeze({
  lifecycle: "ordinary",
  requiredCapabilities: Object.freeze([]),
});

export function runtimeServiceRoute(
  request: RuntimeServiceRequest,
): RuntimeServiceRoute | null {
  if (!isRuntimeServiceMethod(request.service, request.method)) return null;
  return (RUNTIME_SERVICE_ROUTES[request.service] as Readonly<
    Record<string, RuntimeServiceRoute>
  >)[request.method] ?? null;
}

export function runtimeServiceOperationPolicy(
  request: RuntimeServiceRequest,
): RuntimeServiceOperationPolicy {
  const route = runtimeServiceRoute(request);
  if (route === "artifact") return policy("ordinary", "manifold_artifacts");
  if (route === "fitting") {
    return policy("fitting", "fitting", "manifold_artifacts");
  }
  if (request.service === "instruments" && request.method === "startPreparation") {
    const [family, body] = request.args;
    if (isRecord(body)) {
      if (family === "lens" && body.operation === "fit") {
        return policy("ordinary", "jlens_fitting");
      }
      if (family === "sae" && body.operation === "train") {
        return policy("ordinary", "sae_training");
      }
    }
  }
  return ORDINARY;
}

/**
 * Whether this runtime offers a contract method. An `"unavailable"` route is
 * one the worker answers with a refusal (none today: the units members and
 * token anatomy run on hook ABI v5, and a library without unit sites answers
 * native's 422 as drowse does). The client the shared UI holds omits such
 * members, because the UI decides availability by `typeof member ===
 * "function"` (`replayCapabilities`, `unitReadsAvailable`).
 */
export function isOfferedRuntimeMethod(service: RuntimeServiceName, method: string): boolean {
  const route = runtimeServiceRoute({ service, method, args: [] });
  return route !== null && route !== "unavailable";
}

export function isFittingLifecycleRequest(request: RuntimeServiceRequest): boolean {
  return runtimeServiceRoute(request) === "fitting";
}

function policy(
  lifecycle: RuntimeServiceOperationPolicy["lifecycle"],
  ...requiredCapabilities: RuntimeCapabilityOperation[]
): RuntimeServiceOperationPolicy {
  return Object.freeze({
    lifecycle,
    requiredCapabilities: Object.freeze(requiredCapabilities),
  });
}

/** The refusal for a routed method this runtime does not offer (501, not recoverable). */
export function unavailableRuntimeMethod(
  request: Pick<RuntimeServiceRequest, "service" | "method">,
): RuntimeFailure {
  return runtimeFailure(
    "RUNTIME_METHOD_UNAVAILABLE",
    `The browser runtime does not offer ${request.service}.${request.method}`,
    false,
    501,
  );
}

/**
 * The client the shared UI holds: every member of `client` except the
 * `"unavailable"` routes'. The fixture runtime implements those members (as
 * refusals, for the contract inventory) and reaches the UI through this view.
 */
export function offeredRuntimeClient<T extends RuntimeClient>(client: T): T {
  const views = new Map<RuntimeServiceName, object>();
  return new Proxy(client, {
    get(target, property) {
      const value: unknown = Reflect.get(target, property, target);
      if (typeof property !== "string" || !Object.hasOwn(RUNTIME_SERVICE_METHODS, property)) {
        return typeof value === "function" ? value.bind(target) : value;
      }
      const name = property as RuntimeServiceName;
      let view = views.get(name);
      if (!view) {
        view = offeredServiceView(name, value as object);
        views.set(name, view);
      }
      return view;
    },
  });
}

function offeredServiceView(name: RuntimeServiceName, service: object): object {
  const withheld = (method: string | symbol): boolean =>
    typeof method === "string" &&
    isRuntimeServiceMethod(name, method) &&
    !isOfferedRuntimeMethod(name, method);
  return new Proxy(service, {
    get(target, method) {
      if (withheld(method)) return undefined;
      const member: unknown = Reflect.get(target, method, target);
      return typeof member === "function" ? member.bind(target) : member;
    },
    has(target, method) {
      return !withheld(method) && Reflect.has(target, method);
    },
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
