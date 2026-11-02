export const siteName = "Drowse";
export const siteTitle = "Drowse | Local LLM interpretability and activation steering";
export const sourceRepository = "https://github.com/stdlogits/drowse";
export const siteDescription = "Open-source, fully local LLM interpretability workbench in your browser. Inspect token predictions and steer a model's internal activity.";
// The brand colour is Ledger's depth accent where the mark's ramp starts (--deep, light theme).
export const siteAccent = "#6a4fe0";
// Browser chrome follows the page surface, not the accent: Ledger's paper in each theme, as
// theme-init.js and @stdlogits/ledger/tokens.css write it.
export const themeColors = { dark: "#0b0d12", light: "#ffffff" };
export const socialImagePath = "/social/drowse.png?v=ledger-card";
export const socialImageType = "image/png";
export const socialImageAlt = "The drowse mark and name, the steering term 0.4 calm, a reply whose words are tinted by how calm the model reads them, and one word's calm reading across the model's layers.";
export const projectOwner = { "@type": "Person", name: "a9lim", url: "https://github.com/a9lim" };

export function publicOrigin(value = "") {
  if (!value) return "";
  const url = new URL(value);
  if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("DROWSE_PUBLIC_ORIGIN must be an HTTPS origin, such as https://drowse.example, without a path or credentials.");
  }
  return url.origin;
}

const jsonLd = schema => `<script type="application/ld+json">${JSON.stringify(schema).replaceAll("<", "\\u003c")}</script>`;

export function discoveryMetadata(origin, { version } = {}) {
  if (!origin) return "";
  const schema = {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "WebSite", "@id": `${origin}/#website`, url: `${origin}/`, name: siteName, description: siteDescription, inLanguage: "en", publisher: projectOwner },
      { "@type": "WebApplication", "@id": `${origin}/#app`, name: siteName, url: `${origin}/`, description: siteDescription,
        applicationCategory: "DeveloperApplication", operatingSystem: "Web browser", browserRequirements: "WebGPU and a compatible device are required for local model inference.",
        ...(version ? { softwareVersion: version } : {}),
        offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
        author: projectOwner, publisher: projectOwner,
        image: `${origin}${socialImagePath}`, isAccessibleForFree: true,
        license: `${origin}/LICENSE`, sameAs: [sourceRepository],
        featureList: ["Local language model inference", "Activation steering", "Token prediction inspection", "Trait monitoring", "Branching experiments", "R and S lens readouts", "Neuron and head reads and steering", "Compatible SAE packs"] },
    ],
  };
  return `<link rel="canonical" href="${origin}/" />\n<meta property="og:url" content="${origin}/" />\n${jsonLd(schema)}`;
}

/** Canonical, og:url and WebPage JSON-LD for a prerendered page other than the homepage. */
export function pageDiscoveryMetadata(origin, { path, title, description }) {
  if (!origin) return "";
  const url = `${origin}${path}`;
  const schema = {
    "@context": "https://schema.org",
    "@type": "WebPage", "@id": `${url}#page`, url, name: title, description, inLanguage: "en",
    isPartOf: { "@id": `${origin}/#website` }, about: { "@id": `${origin}/#app` },
  };
  return `<link rel="canonical" href="${url}" />\n<meta property="og:url" content="${url}" />\n${jsonLd(schema)}`;
}
