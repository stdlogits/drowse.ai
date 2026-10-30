# Security Policy

This policy covers the drowse.ai site: the hosted shell, the browser runtime,
artifact verification, and the `/api/contact` Worker. For the Python package,
`drowse serve`, and model or checkpoint loading, see the drowse
[security policy](https://github.com/stdlogits/drowse/security/policy).

## Reporting a vulnerability

Please report security issues privately rather than filing a public issue.

- **GitHub:** use [private security advisories](https://github.com/stdlogits/drowse.ai/security/advisories/new)
- **Email:** [contact@drowse.ai](mailto:contact@drowse.ai)

Please include a description, reproduction steps, the page or build revision
(the `drowse-source-revision` meta tag), the browser and GPU, and the model
involved. Do not include private prompts or credentials in the report.

## Supported versions

Only the build currently deployed at drowse.ai receives fixes.

## Browser data

Hosted conversations, fitted artifacts, and model downloads are stored locally
in browser storage. They are not encrypted by Drowse. The service worker caches
application assets; it does not cache conversation API responses. Published SAE
descriptions for the verified Gemma Scope 2 packs ship as application assets.
Missing descriptions fall back to Neuronpedia, sending only the public
dictionary/feature identity without cookies, referrers, activation values, or
conversation text.

"Clear all local data" stops persistence and clears the application's saved
sessions, artifacts, and preferences, including legacy Polythetic/Saklas
databases and files that migration could otherwise restore on reload. Legacy
directory handles remain usable after their contents are removed. A failed deletion is reported and can be
retried; it must not silently report success. Browser profiles, extensions, OS
accounts, and external backups remain outside this deletion boundary.
