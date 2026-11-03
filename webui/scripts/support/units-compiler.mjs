// The structured compiler over the tiny fixture model, as the model load would set it up: a core
// pack holding the fixture session's neutral whitener (and the `deceptive_fixture` concept
// drowse registers in Python for unit-ablation's `mixed` case), and the units service's program
// source over the reference backend and B6-shaped packs. `compile` prepares first, as
// `compileSteering` does.
import { readFileSync } from "node:fs";
import { corePackRequest } from "./core-pack.mjs";
import { unitsPackSource } from "./units-fixture.mjs";

export const fixtureWhiteners = JSON.parse(readFileSync(new URL("../../../browser-runtime/fixtures/units-tiny-whitener-v1.json", import.meta.url), "utf8"));

export async function unitsCompiler(server, fx, arch) {
  const [{ BrowserFeasibilityCorePackCompiler }, service] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/browserCorePack.ts"),
    server.ssrLoadModule("/src/hosted/runtime/units/unitsService.ts"),
  ]);
  const d = fx.model.hiddenSize;
  const unit = (n) => Array.from({ length: d }, (_, i) => (i === n ? 1 : 0));
  const request = await corePackRequest(server, {
    hiddenSize: d,
    layerCount: 4,
    modelId: "units-tiny",
    whiteners: fixtureWhiteners.arches[arch].map((row) => ({ layer: row.layer, mean: row.mean, basis: row.basis, eigenvalues: row.eigenvalues, ridge: row.ridge })),
    manifolds: [{
      name: "deceptive_fixture",
      labels: ["deceptive_fixture_pole", "deceptive_fixture_anti"],
      nodeCoordinates: [1, -1],
      layers: [0, 1, 2, 3].map((layer) => ({ layer, basis: [unit(layer)], nodeCoordinates: [[1], [-1]] })),
    }],
  });
  const compiler = await BrowserFeasibilityCorePackCompiler.load(request);
  const units = new service.BrowserUnitsService({
    modelId: "units-tiny",
    blockSites: () => fx.backend,
    packs: () => unitsPackSource(fx, arch, fixtureWhiteners),
    tokenTexts: async (ids) => ids.map((id) => fx.tokenizer.decode([id])),
    resolveToken: async (word) => fx.tokenizer.encode(word)[0],
    replay: () => null,
  });
  compiler.setUnitSource(await units.programSource());
  return {
    compiler,
    units,
    async compile(expression) {
      await compiler.prepareUnitSelectors(expression);
      return compiler.compile(expression);
    },
  };
}
