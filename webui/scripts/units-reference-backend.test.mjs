// The test-only CPU reference backend (webui/src/hosted/runtime/units/): the tiny fixture
// models regenerate to the model file's digests, and the dense forward reproduces drowse's
// block-sites goldens (every site, layer and position; the last logits).
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { ARCHES, archGolden, assertClose, decode, golden, viteServer } from "./support/units-goldens.mjs";

const server = await viteServer();
try {
  const tiny = await server.ssrLoadModule("/src/hosted/runtime/units/tinyModels.ts");
  const dense = await server.ssrLoadModule("/src/hosted/runtime/units/denseModel.ts");
  const file = golden("units-tiny-models-v1.json");

  for (const arch of ARCHES) {
    const record = file.arches[arch];
    const tensors = tiny.tinyTensors(file, arch);
    for (const entry of record.tensors) {
      const value = tensors.get(entry.name);
      const digest = createHash("sha256").update(Buffer.from(value.buffer, value.byteOffset, value.byteLength)).digest("hex");
      assert.equal(digest, entry.sha256, `${arch} ${entry.name} regenerates to its digest`);
    }
    const tokenizer = tiny.tinyTokenizer(record);
    assert.equal(tokenizer.size, record.tokenizer.size);
    const text = { gemma3: "Kp=q;Wx7&zR", qwen3: "Kp=q;Wx7&zR!" }[arch];
    const ids = [...(record.tokenizer.bos === null ? [] : [record.tokenizer.bos]), ...tokenizer.encode(text)];
    assert.deepEqual(ids, record.short_prompt, `${arch} short prompt tokenizes as drowse's`);
    assert.equal(tokenizer.decode(record.short_prompt), (record.tokenizer.bos === null ? "" : "<bos>") + text);

    // ------------------------------------------------------- block sites (B1-goldens §2.1) ---
    const sites = archGolden("block-sites", arch);
    const tol = sites.tolerance.library;
    const model = tiny.tinyModelWeights(file, arch);
    const sequence = new dense.DenseSequence(model, 64);
    const captured = {};
    const hidden = sequence.forward(sites.ids, {
      capture(layer, site) {
        for (const [name, key] of [
          ["block_in", "blockIn"], ["z", "z"], ["attn_o", "attnO"], ["mid", "mid"],
          ["act", "act"], ["mlp_o", "mlpO"], ["out", "out"],
        ]) (captured[name] ??= [])[layer] = site[key];
      },
    });
    for (const [name, value] of Object.entries(sites.sites)) {
      const expected = decode(value);
      const actual = new Float32Array(expected.length);
      let at = 0;
      for (const layer of captured[name]) {
        actual.set(layer, at);
        at += layer.length;
      }
      assertClose(actual, expected, tol, `${arch} block-sites ${name}`);
    }
    const last = (sites.ids.length - 1) * model.hiddenSize;
    assertClose(dense.finalLogits(model, hidden, last), decode(sites.logits_last), tol, `${arch} last logits`);
  }
  console.log("units reference backend: tiny models and block sites match drowse");
} finally {
  await server.close();
}
