import type {
  BrowserFittingCentroids,
  BrowserFittingTopologyFoundation,
  FinalizedBrowserAffineLayer,
} from "./coordinator";
import { applySerializedInverse } from "./mahalanobis";
import type { SerializedMahalanobisWhitener, TopologyResult } from "./workerContracts";

/**
 * A one-node ``pca`` concept against the neutral baseline, as drowse's ``_fit_monopolar``
 * folds it (``fold_directions_to_subspace``): per layer the direction d is the concept centroid
 * minus the neutral mean; the frame writes along d/‖d‖_M and reads with Σ⁻¹d/‖d‖_M, the mean is
 * the neutral mean projected into the span through the read row, the pole sits at ‖d‖_M (so at
 * ν + d), and the share is ‖d‖_M. No discover coordinates, no DLS: every evaluated layer folds
 * verbatim.
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
    if (Math.hypot(...direction) <= 1e-12) continue;
    // A typed, Mahalanobis-unit frame: write along d, read with its covector Σ⁻¹d.
    const inverseDirection = applySerializedInverse(whitener, direction);
    let squared = 0;
    for (let index = 0; index < pooled.columns; index += 1) {
      squared += direction[index] * inverseDirection[index];
    }
    const share = Math.sqrt(Math.max(squared, 0));
    if (!Number.isFinite(share) || share <= 1e-12) continue;
    const basis = Float64Array.from(direction, (value) => value / share);
    const read = Float64Array.from(inverseDirection, (value) => value / share);
    let neutralCoordinate = 0;
    for (let index = 0; index < pooled.columns; index += 1) {
      neutralCoordinate += whitener.mean[index] * read[index];
    }
    const mean = Float64Array.from(basis, (value) => value * neutralCoordinate);
    layers.set(layer, {
      operation: "affine_fisher",
      nodeCount: 1,
      columns: pooled.columns,
      components: 1,
      centroidMean: pooled.values.slice(),
      mean,
      basis,
      read,
      // The pole sits at ν + d: ‖d‖_M along the Mahalanobis-unit write row.
      nodeCoordinates: new Float64Array([share]),
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
