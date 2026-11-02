// Where the hosted catalog is published. Each browser runtime generation publishes its signed
// catalog at its own directory of the catalog repository, so publishing a new catalog never
// changes what an already deployed build reads: runtime v1 builds read the root `catalog.json`
// (sequences up to 10), runtime v2 builds (hook ABI v5) read `v2/catalog.json` (from 11). The
// distribution lock names the newest directory; the publisher's first catalog at a new directory
// must advance the sequence the previous one published.

export const CATALOG_REPOSITORY = "stdlogits/drowse-web-catalog";

/** Every directory the catalog has been published at, oldest first. */
export const CATALOG_DIRECTORIES = Object.freeze(["", "v2/"]);

export function catalogUrls(directory, repository = CATALOG_REPOSITORY) {
  const base = `https://huggingface.co/${repository}/resolve/main/${directory}`;
  return { catalogUrl: `${base}catalog.json`, signatureUrl: `${base}catalog.sig.json` };
}

/** The URLs the distribution lock must name: the newest directory's. */
export const CURRENT_CATALOG_URLS = Object.freeze(catalogUrls(CATALOG_DIRECTORIES.at(-1)));

/** The repository and directory a distribution lock's catalog and signature publish to; both in
 *  one directory of one repository, and that directory one the catalog has a lineage for. */
export function catalogLocation(distribution) {
  const catalog = parseCatalogUrl(distribution.catalogUrl, "catalogUrl", "catalog.json");
  const signature = parseCatalogUrl(distribution.signatureUrl, "signatureUrl", "catalog.sig.json");
  if (catalog.repository !== signature.repository || catalog.directory !== signature.directory) {
    throw new Error("catalog and signature must publish to one directory of one repository");
  }
  if (!CATALOG_DIRECTORIES.includes(catalog.directory)) {
    throw new Error(`catalog directory ${JSON.stringify(catalog.directory)} is not a published catalog path`);
  }
  return catalog;
}

/** The directories published before `directory`, newest first. */
export function previousCatalogDirectories(directory) {
  const index = CATALOG_DIRECTORIES.indexOf(directory);
  if (index < 0) throw new Error(`catalog directory ${JSON.stringify(directory)} is not a published catalog path`);
  return CATALOG_DIRECTORIES.slice(0, index).reverse();
}

function parseCatalogUrl(value, label, filename) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${label} is invalid`);
  }
  const parts = url.pathname.split("/").filter(Boolean);
  if (
    url.origin !== "https://huggingface.co" || url.username || url.password || url.search || url.hash ||
    parts.length < 5 || parts[2] !== "resolve" || parts[3] !== "main" || parts.at(-1) !== filename ||
    parts.slice(4, -1).some((part) => part === "." || part === "..")
  ) {
    throw new Error(`${label} is not a supported Hugging Face main-branch URL`);
  }
  const directory = parts.slice(4, -1).map((part) => `${part}/`).join("");
  return { repository: `${parts[0]}/${parts[1]}`, directory };
}
