import type {
  BrowserFittingCentroids,
  BrowserFittingTopologyFoundation,
  FinalizedBrowserAffineLayer,
} from "./coordinator";
import { mahalanobisNorm } from "./mahalanobis";
import type { SerializedMahalanobisWhitener, TopologyResult } from "./workerContracts";

/**
 * A one-node ``pca`` concept against the neutral baseline, as drowse's ``_fit_monopolar``
 * folds it (``fold_directions_to_subspace``): per layer the direction is the concept centroid
 * minus the neutral mean, the basis its unit vector, the mean the neutral mean projected onto
 * it, the pole one step of the direction's norm from the neutral origin, and the share its
 * whitened norm. No discover coordinates, no DLS: every evaluated layer folds verbatim.
 * The in-app fit and the core-pack builder both take this path.
 */
export function monopolarFoundation(
  centroids: BrowserFittingCentroids,
  whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>,
): BrowserFittingTopologyFoundation {
  const layers = new Map<number, FinalizedBrowserAffineLayer>();
  for (const [layer, pooled] of centroids.layers) {
    const whitener = whiteners.get(layer);
    if (!whitener || pooled.rows !== 1 || pooled.columns !== whitener.columns) {
      throw fittingError(
        "BROWSER_FIT_RESULT_INVALID",
        `Browser monopolar centroid for layer ${layer} does not match its neutral whitener`,
        500,
      );
    }
    const direction = Float64Array.from(
      pooled.values,
      (value, index) => value - whitener.mean[index],
    );
    const norm = Math.hypot(...direction);
    if (norm <= 1e-12) continue;
    const basis = Float64Array.from(direction, (value) => value / norm);
    let neutralCoordinate = 0;
    for (let index = 0; index < pooled.columns; index += 1) {
      neutralCoordinate += whitener.mean[index] * basis[index];
    }
    const mean = Float64Array.from(basis, (value) => value * neutralCoordinate);
    const share = mahalanobisNorm(whitener, direction);
    if (!Number.isFinite(share) || share <= 0) continue;
    layers.set(layer, {
      operation: "affine_fisher",
      nodeCount: 1,
      columns: pooled.columns,
      components: 1,
      centroidMean: pooled.values.slice(),
      mean,
      basis,
      nodeCoordinates: new Float64Array([norm]),
      muCoordinates: new Float64Array([0]),
      whitenedGram: new Float64Array([share * share]),
      neutralCrossGram: new Float64Array([0]),
      explainedVariance: 1,
      mahalanobisShare: share,
      affineMap: null,
    });
  }
  if (layers.size === 0) {
    throw fittingError(
      "BROWSER_MONOPOLAR_DEGENERATE",
      "The concept activation is indistinguishable from the neutral baseline on every fitted layer",
    );
  }
  const topology: TopologyResult = {
    operation: "topology",
    winnerName: "monopolar-neutral-ray",
    fitMode: "pca",
    intrinsicDimensions: 1,
    periodicDimensions: 0,
    persistentLoops: 0,
    usedFaintCycle: false,
    coordinates: new Float64Array([1]),
    embeddedCoordinates: new Float64Array([1]),
    candidates: [],
    diagnostics: {
      kind: "pca",
      perComponentVariance: new Float64Array([1]),
      cumulativeVariance: new Float64Array([1]),
      pickedDimensions: 1,
      threshold: 0.7,
    },
    winnerPlan: null,
  };
  return {
    identity: centroids.identity,
    layers,
    captureRetained: centroids.captureRetained,
    consensusGram: new Float64Array([0]),
    topology,
    dlsKept: null,
    neutralLayoutCoordinate: new Float64Array([0]),
    anchoredNodeCoordinates: new Float64Array([1]),
    finalAffineLayers: layers,
    curvedSurfaceLayers: null,
  };
}

function fittingError(code: string, message: string, status = 409): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status, recoverable: status < 500 });
}
