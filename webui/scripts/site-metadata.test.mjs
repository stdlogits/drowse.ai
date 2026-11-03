import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { siteDescription, siteAccent, themeColors, socialImagePath, socialImageType, socialImageAlt, publicOrigin, discoveryMetadata, pageDiscoveryMetadata, projectOwner } from "./site-metadata.mjs";

function readJsonLd(html) {
  const scriptStart = '<script type="application/ld+json">';
  assert.ok(html.includes(scriptStart));
  assert.ok(html.endsWith("</script>"));
  return JSON.parse(html.slice(html.indexOf(scriptStart) + scriptStart.length, -"</script>".length));
}

test("public metadata requires an explicit, safe HTTPS origin", () => {
  assert.equal(publicOrigin(), "");
  assert.equal(discoveryMetadata(""), "");
  assert.equal(publicOrigin("https://drowse.example/"), "https://drowse.example");
  for (const input of ["http://example.com", "https://user:pass@example.com", "https://example.com/app", "https://example.com/?q=x", "https://example.com/#x"]) assert.throws(() => publicOrigin(input));
});

test("structured data describes only the website and actual application", () => {
  const html = discoveryMetadata("https://drowse.example", { version: "5.3.0" });
  const graph = readJsonLd(html)["@graph"];
  assert.deepEqual(graph.map(item => item["@type"]), ["WebSite", "WebApplication"]);
  assert.ok(graph.every(item => item.name === "Drowse" && item.description === siteDescription));
  assert.ok(graph.every(item => !item.aggregateRating && !item.potentialAction));
  assert.equal(graph[1].image, `https://drowse.example${socialImagePath}`);
  assert.equal(graph[1].url, "https://drowse.example/");
  assert.equal(graph[1].softwareVersion, "5.3.0");
  assert.deepEqual(graph[1].offers, { "@type": "Offer", price: "0", priceCurrency: "USD" });
  assert.deepEqual(graph[1].author, projectOwner);
  assert.deepEqual(graph[1].publisher, { "@type": "Person", name: "a9lim", url: "https://github.com/a9lim" });
  assert.ok(!("screenshot" in graph[1]), "no product screenshot asset exists to publish");
  assert.ok(!("softwareVersion" in readJsonLd(discoveryMetadata("https://drowse.example"))["@graph"][1]));
  assert.match(html, /rel="canonical" href="https:\/\/drowse.example\/"/);
});

test("prerendered pages get their own canonical URL and WebPage data", () => {
  assert.equal(pageDiscoveryMetadata("", { path: "/privacy", title: "t", description: "d" }), "");
  const html = pageDiscoveryMetadata("https://drowse.example", { path: "/privacy", title: "Privacy <Drowse>", description: "Local data" });
  assert.match(html, /<link rel="canonical" href="https:\/\/drowse.example\/privacy" \/>/);
  assert.match(html, /<meta property="og:url" content="https:\/\/drowse.example\/privacy" \/>/);
  assert.doesNotMatch(html, /<Drowse>/, "JSON-LD must not open markup");
  const page = readJsonLd(html);
  assert.equal(page["@type"], "WebPage");
  assert.equal(page.name, "Privacy <Drowse>");
  assert.deepEqual(page.isPartOf, { "@id": "https://drowse.example/#website" });
});

test("SEO and social metadata use the requested description and Ledger's colours", async () => {
  assert.equal(siteDescription, "Open-source, fully local LLM interpretability workbench in your browser. Inspect token predictions and steer a model's internal activity.");
  assert.ok(siteDescription.length <= 160, "search snippets truncate longer descriptions");
  assert.doesNotMatch(siteDescription, /an LLMs|fully[^.]*fully/);
  assert.ok((await readFile("../drowse/webui/index.html", "utf8")).includes(`name="description" content="${siteDescription}"`), "the Python dashboard's index.html must use the shared description");
  assert.ok((await readFile("hosted/index.html", "utf8")).includes('content="__DROWSE_HOSTED_DESCRIPTION__"'));
  // The brand colour is the depth accent where the mark's ramp starts; the browser chrome is paper.
  const tokens = await readFile("../node_modules/@stdlogits/ledger/src/styles/tokens.css", "utf8");
  const light = tokens.slice(tokens.indexOf(":root {"), tokens.indexOf("}", tokens.indexOf(":root {")));
  const dark = tokens.slice(tokens.indexOf(':root[data-theme="dark"] {'), tokens.indexOf("}", tokens.indexOf(':root[data-theme="dark"] {')));
  const token = (body, name) => body.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6});`, "i"))?.[1].toLowerCase();
  assert.equal(siteAccent, token(light, "--deep"));
  assert.deepEqual(themeColors, { light: token(light, "--paper"), dark: token(dark, "--paper") });
  assert.ok((await readFile("../drowse/webui/public/favicon.svg", "utf8")).includes(`stop-color: ${siteAccent}`), "the mark's ramp starts at the brand colour");
  // The link card is drowse's, byte for byte: drawn in Ledger by drowse's scripts/social-card.mjs, whose
  // HTML carries the alt text. The site serves the PNG (a flat card is smaller and sharper as PNG).
  assert.equal(socialImagePath, "/social/drowse.png?v=ledger-card");
  assert.equal(socialImageType, "image/png");
  const card = await readFile("../drowse/webui/scripts/social-card.html", "utf8");
  const cardAlt = card.match(/Alt text: "([^"]+)"/)?.[1].replace(/\s+/g, " ");
  assert.equal(socialImageAlt, cardAlt, "the alt text is the card's own");
  await assert.rejects(readFile("public-hosted/social/drowse.jpg"), { code: "ENOENT" }, "one share image");
  assert.deepEqual(await readFile("../drowse/webui/public/social/drowse.png"), await readFile("public-hosted/social/drowse.png"));
});

test("social preview and icon assets have the declared PNG dimensions", async () => {
  const dimensions = async path => {
    const bytes = await readFile(path);
    assert.equal(bytes.subarray(1, 4).toString(), "PNG");
    return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  };
  for (const root of ["../drowse/webui/public", "public-hosted"]) {
    assert.deepEqual(await dimensions(`${root}/social/drowse.png`), [1200, 630]);
    assert.deepEqual(await dimensions(`${root}/icons/apple-touch-icon.png`), [180, 180]);
    assert.deepEqual(await dimensions(`${root}/icons/drowse-192.png`), [192, 192]);
    assert.deepEqual(await dimensions(`${root}/icons/drowse-512.png`), [512, 512]);
    assert.deepEqual(await dimensions(`${root}/icons/drowse-maskable-512.png`), [512, 512]);
  }
  // The brand set is drowse's, byte for byte: the closed eye on the depth ramp.
  for (const file of ["favicon.svg", "favicon.ico", "icons/apple-touch-icon.png", "icons/drowse-192.png", "icons/drowse-512.png", "icons/drowse-maskable-512.png"]) {
    assert.deepEqual(await readFile(`public-hosted/${file}`), await readFile(`../drowse/webui/public/${file}`), `${file} matches drowse`);
  }
  const ico = await readFile("public-hosted/favicon.ico");
  assert.deepEqual([ico.readUInt16LE(0), ico.readUInt16LE(2)], [0, 1], "favicon.ico is an icon resource");
  const sizes = Array.from({ length: ico.readUInt16LE(4) }, (_, i) => ico[6 + i * 16] || 256);
  assert.ok(sizes.includes(16) && sizes.includes(32), `favicon.ico carries 16 and 32 px (has ${sizes.join(", ")})`);
  const index = await readFile("hosted/index.html", "utf8");
  for (const link of ['<link rel="icon" href="/favicon.ico" sizes="32x32" />', '<link rel="icon" type="image/svg+xml" href="/favicon.svg" />', '<link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />']) {
    assert.ok(index.includes(link), `hosted index links ${link}`);
  }
  await assert.rejects(readFile("public-hosted/icons/tab-home.png"), { code: "ENOENT" }, "tab icons are gone with tab identity");
});
