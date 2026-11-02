# Hosted instrument release corpora

## The native lens corpus

Hosted R and S lenses are drowse's native fits over the pinned FineWeb-Edu corpus that
`native-lens-corpus.json` records (`browser-runtime/README.md`, "Lens, units and core
packs"). Its `attributionApproval` is
`{"reviewStatus": "approved", "reviewedBy", "reviewedAt", "attribution"}` (a9lim approved it on
2026-11-02); the release policy (`hosted-instrument-release-policy.mjs`) refuses every native R,
S and units pack while it is null or incomplete. The `attribution` text travels with everything
published from these fits:

- `build_hosted_packs.py` writes it, with the dataset URL, license and corpus spec, into every
  R, S and units pack as `packs/<kind>/ATTRIBUTION.md`, and refuses to build those packs from an
  unapproved record; the release policy requires the file and its text, and publishing and
  catalog admission run that policy, so the notice reaches the `-instruments` repositories on
  Hugging Face and every browser that installs a pack.
- The site's Credits page lists FineWeb-Edu under "Interpretability tools and data"
  (`webui/src/hosted/ui/Credits.svelte`; `credits.test.mjs` ties it to this record).

`jlens-word-seeds.txt` is the R pack's curated browser vocabulary (`build_hosted_packs.py`).

## The SAE release corpus

`prepare_hosted_release_corpus.py` produces the bounded JSON input of the hosted SAE builder,
and only that: the lenses are native fits on the corpus above. It is offline and fail closed:
the exact source snapshot and pinned license evidence must match the source manifest
(`release-corpus-source.schema.json`, schema 2) before it writes anything. Gemma uses Gemma
Scope SAE packs rather than this local fitting corpus; GPT-2 and Pythia keep their provider R
lenses.

An approved run emits:

- `sae-corpus.json`
- `release-corpus.json`, which records the input and output digests, source and license
  provenance, selection limits and the generator digest; `release-corpus.schema.json`
  (schema 2) is its public contract

The generator normalizes Unicode to NFC, collapses whitespace, preserves source row order, packs
only bounded documents, and writes canonical UTF-8 JSON. It refuses an existing output
directory. Parquet input requires `pyarrow`.

Schema 1 also carried `jlens-corpus.json`, `jlens-words.json`, word seeds and release
tokenizers for the retired capture-based R lens builder; schema 2 drops them, and the policy
refuses a manifest that still names them.

## Sources

`fineweb-sample-10bt-train.approved.json` is the approved SAE corpus source (license review of
2026-08-30). Schema 2 removed its word-seed and tokenizer fields and renamed its selection
algorithm to `drowse-release-corpus-v1`; the source, license and review are unchanged.

`fineweb-sample-10bt-train.candidate.json` is the same shard before that review. It pins one 574,962,194-byte train shard and the exact dataset card at FineWeb commit
`9bb295ddab0e05d785b879661af7260fed5140fc`, and its file digest, 281,664-row Parquet metadata,
ODC-By declaration and Common Crawl terms notice are closed. The Parquet reader streams row
groups and stops after the bounded 320-document output. It stays blocked; do not infer legal
approval from the dataset card.

`wikitext-103-raw-test.candidate.json` records the exact cached WikiText shard, but it is not a
production manifest. The pinned dataset card's YAML license list (`cc-by-sa-3.0`, `gfdl`)
conflicts with its prose (`CC BY-SA 4.0`), the cached file is the held-out test split, and no
reviewed redistribution notice bundle is available. The generator exits before touching the
inputs while those blockers remain.

Do not change `status` by itself. A production manifest needs an explicitly approved non-test
snapshot, an exact license-evidence file, a completed attribution/redistribution notice, a named
reviewer and review timestamp, and no remaining blockers. Legal review is an external release
requirement; this tool enforces its recorded result but does not make it.

## Approved run

```bash
python3 webui/scripts/prepare_hosted_release_corpus.py \
  browser-runtime/release-corpus/fineweb-sample-10bt-train.approved.json \
  /immutable/source/shard.parquet \
  /immutable/source/README.md \
  /output/drowse-release-corpus
```

Pass `sae-corpus.json` to `build-hosted-sae.mjs` (`--release-corpus`, `--release-corpus-source`,
`--release-license-evidence`). Preserve `release-corpus.json` with the SAE pack.
