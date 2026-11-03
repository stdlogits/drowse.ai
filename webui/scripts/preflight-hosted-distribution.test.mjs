import assert from "node:assert/strict";
import {
  assertCatalogSequenceFloor,
  fetchChecked,
  validatedApprovedUrl,
} from "./preflight-hosted-distribution.mjs";

const cors = (body, { status = 200, headers = {}, origin = "*" } = {}) =>
  new Response(body, { status, headers: { "Access-Control-Allow-Origin": origin, ...headers } });
const redirect = (location) => cors(null, { status: 302, headers: { Location: location } });

// Follows approved redirects, relative ones included, as a credential-free browser request.
{
  const calls = [];
  const response = await fetchChecked("https://source.test/file", new Set(["https://cdn.test"]), 1_000, async (url, init) => {
    calls.push({ url, init });
    if (url === "https://source.test/file") return redirect("https://cdn.test/first");
    if (url === "https://cdn.test/first") return cors(null, { status: 307, headers: { Location: "/final" } });
    return cors(Uint8Array.of(1));
  });
  assert.equal(response.status, 200);
  assert.deepEqual(calls.map(({ url }) => url), ["https://source.test/file", "https://cdn.test/first", "https://cdn.test/final"]);
  for (const { init } of calls) {
    assert.equal(init.redirect, "manual");
    assert.equal(init.credentials, "omit");
  }
}

// Refuses an unapproved hop before requesting it, a sixth redirect, and insecure or credentialed URLs.
{
  let calls = 0;
  await assert.rejects(
    fetchChecked("https://source.test/file", new Set(["https://cdn.test"]), 1_000, async () => {
      calls += 1;
      return redirect("https://unapproved.test/file");
    }),
    /redirected to unapproved URL/,
  );
  assert.equal(calls, 1);
  let hops = 0;
  await assert.rejects(
    fetchChecked("https://source.test/file", new Set(["https://cdn.test"]), 1_000, async () => {
      hops += 1;
      return redirect(`https://cdn.test/hop-${hops}`);
    }),
    /exceeded five approved redirects/,
  );
  assert.equal(hops, 6);
  const never = async () => assert.fail("fetched an unapproved URL");
  await assert.rejects(fetchChecked("http://source.test/file", new Set(), 1_000, never), /unapproved URL/);
  assert.throws(
    () => validatedApprovedUrl("https://user:secret@source.test/file", new Set(["https://source.test"]), "https://source.test/file"),
    /redirected to unapproved URL/,
  );
}

// Requires the CORS answer a browser needs.
await assert.rejects(
  fetchChecked("https://source.test/file", new Set(), 1_000, async () => cors(null, { origin: "https://other.test" })),
  /did not allow the requested browser origin/,
);

assert.doesNotThrow(() => assertCatalogSequenceFloor(8, 7));
assert.throws(() => assertCatalogSequenceFloor(6, 7), /below the embedded rollback floor/);

console.log("Hosted catalog preflight tools passed");
