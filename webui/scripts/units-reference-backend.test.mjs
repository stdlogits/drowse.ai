// The fixture runtime's tiny models (webui/src/hosted/runtime/units/): each arch's weights
// regenerate to the model file's digests and its tokenizer reads the recorded short prompt.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const tiny = await server.ssrLoadModule("/src/hosted/runtime/units/tinyModels.ts");
  const file = JSON.parse(readFileSync(new URL("../src/hosted/runtime/fixtures/units-tiny-models-v1.json", import.meta.url), "utf8"));

  for (const arch of ["gemma3", "qwen3"]) {
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
  }
  console.log("units tiny models: weights and tokenizers regenerate");
} finally {
  await server.close();
}
