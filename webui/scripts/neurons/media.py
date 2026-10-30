"""Fetch and pack the images and sounds Gemma 4's poles fire on most, from the collections whose
licences let this site show them, as the model saw them: 384 × 384 images (photographs
centre-cropped, documents and drawings letterboxed) and the middle 2.56 s of each sound at 16 kHz.

    uv run --no-project --with-editable "$HOME/Work/neuron-lens[media]" python -I \
        scripts/neurons/media.py <catalog dir> <out dir> [--sources fleurs,mnist] [--modality audio]

``--modality ""`` only rewrites the credits from the packs already on disk.

The catalog dir is neuron-lens's ``data/catalog/gemma-4-12b``; the out dir is the model's data folder
(``public-hosted/neurons-data/gemma-4-12b``). Each allowed source is streamed again with
neuron-lens's own reader (``media_sets._take``: the same split, seed and count as the exemplar run),
and an item is kept only when its bytes' SHA-1 matches its document's, so a page never shows an image
or sound other than the one the catalog names. Synthetic probes are drawn again and matched by their
labels in order; photographs degraded from COCO stay out.

Output, under ``<out dir>/media/``:

- ``<modality>/p<pack>.bin``: WebP thumbnails (192 px) or MP3 clips (32 kbps), end to end, each pack
  under ``PACK_BYTES``; the page fetches one by HTTP range (one file a pack keeps the site under
  Cloudflare Pages' file limit);
- ``<modality>/hosted.json``: document -> [pack, offset, length];
- ``credits.json``: each shown source's name, licence and credit, for the page.

``build-neuron-catalog.py`` folds ``hosted.json`` into the media document files: run it after.
Collections not listed here (ImageNet, AudioSet's YouTube clips, CelebA and others whose terms don't
allow redistribution) are shown by their labels and linked to their source.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# source -> (name, licence, credit). Licences as their publishers state them; the site is non-commercial.
ALLOWED = {
    "image": {
        "quickdraw": ("Quick, Draw!", "CC BY 4.0", "Google Creative Lab"),
        "mnist": ("MNIST", "CC BY-SA 3.0", "Yann LeCun and Corinna Cortes"),
        "fashionmnist": ("Fashion-MNIST", "MIT", "Zalando Research"),
        "eurosat": ("EuroSAT", "MIT", "Helber et al.; contains modified Copernicus Sentinel data"),
        "xray": ("Chest X-ray images", "CC BY 4.0", "Kermany, Zhang and Goldbaum"),
        "histology": ("PatchCamelyon", "CC0 1.0", "Veeling et al."),
        "pubtabnet": ("PubTabNet", "CDLA-Permissive 1.0", "IBM; tables from PubMed Central open-access articles"),
        "websight": ("WebSight", "CC BY 4.0", "Hugging Face"),
        "rico": ("RICO Screen2Words", "CC BY 4.0", "Wang et al., Google Research"),
        "aigen": ("Synthetic DALL·E 3 images", "MIT", "ProGamerGov"),
        "im2latex": ("im2latex-100k", "CC0 1.0", "Deng et al.; rendered from arXiv LaTeX"),
        "skin": ("HAM10000", "CC BY-NC-SA 4.0", "Tschandl, Rosendahl and Kittler"),
        "galaxy": ("Galaxy10 DECaLS", "public data", "Leung and Bovy; DESI Legacy Imaging Surveys"),
        "aerial": ("UC Merced Land Use", "public domain", "Yang and Newsam; USGS National Map imagery"),
        "underwater": ("Aquarium", "CC BY 4.0", "Roboflow"),
        "synth": ("Synthetic probes", "CC BY 4.0", "neuron-lens"),
    },
    "audio": {
        "fleurs": ("FLEURS", "CC BY 4.0", "Conneau et al., Google"),
        "mls": ("Multilingual LibriSpeech", "CC BY 4.0", "Pratap et al., Meta; LibriVox recordings"),
        "voxpopuli": ("VoxPopuli", "CC0 1.0", "Wang et al., Meta; European Parliament"),
        "edacc": ("EdAcc", "CC BY-SA 4.0", "Sanabria et al., University of Edinburgh"),
        "expresso": ("Expresso", "CC BY-NC 4.0", "Nguyen et al., Meta"),
        "esc50": ("ESC-50", "CC BY-NC 3.0", "Karol J. Piczak; clips from Freesound"),
        "synth": ("Synthetic probes", "CC BY 4.0", "neuron-lens"),
    },
}
THUMB = 192
MP3_KBPS = 32
PACK_BYTES = 16_000_000
SCAN_SECONDS = 1800  # a source's file-order scan stops after this


def encode_image(im) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    im.resize((THUMB, THUMB), Image.LANCZOS).save(buf, "WEBP", quality=76, method=6)
    return buf.getvalue()


def encode_audio(w: np.ndarray) -> bytes:
    pcm = (np.clip(w, -1, 1) * 32767).astype("<i2").tobytes()
    run = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "-",
                          "-c:a", "libmp3lame", "-b:a", f"{MP3_KBPS}k", "-f", "mp3", "-"], input=pcm, capture_output=True, check=True)
    return run.stdout


def fetch(modality: str, sources: set[str], docs: list[dict]) -> dict[int, bytes]:
    """doc -> encoded bytes, for every document of an allowed source whose item streams back identical."""
    from neuron_lens import media_sets as ms

    by_sha, doc_source = {}, {}
    synth_docs = []
    for r in docs:
        if r["source"] not in sources:
            continue
        doc_source[r["doc"]] = r["source"]
        ref = json.loads(r["ref"]) if r["ref"] else {}
        if r["source"] == "synth":
            if "degraded" not in ref:
                synth_docs.append(r)
        elif ref.get("sha1"):
            by_sha[ref["sha1"]] = r["doc"]
    encode = encode_image if modality == "image" else encode_audio
    out = {}

    jobs = [j for j in (ms.image_sources(1.0) if modality == "image" else ms.audio_sources(1.0)) if j[0] in sources]

    def take(j):
        """The exemplar run's own sample first; then, for what a different shuffle order missed, the
        split in file order, matching each row's bytes, until every document is found or SCAN_SECONDS."""
        source, name, cfg, split, n, field = j[:6]
        buffer = j[8] if len(j) > 8 else None
        t0 = time.time()
        items = ms._take(*j[:8], modality, buffer=buffer)
        got = {by_sha[x[4]["sha1"]]: encode(x[0]) for x in items if x[4].get("sha1") in by_sha}
        want = {sha: doc for sha, doc in by_sha.items() if doc_source[doc] == source and doc not in got}
        first = len(got)
        if want and buffer != "folder":
            from datasets import load_dataset
            letterbox = source in ms.LETTERBOX
            try:
                for row in load_dataset(name, cfg, split=split, streaming=True).decode(False):
                    data = row[field]
                    sha = hashlib.sha1(ms._bytes(data)).hexdigest()
                    if sha in want:
                        x = ms._image(data, letterbox) if modality == "image" else ms._clip(data)
                        if x is not None:
                            got[want.pop(sha)] = encode(x)
                    if not want or time.time() - t0 > SCAN_SECONDS:
                        break
            except Exception as e:  # one failing source must not sink the rest
                print(f"{modality} {source}: scan failed {e!r}"[:200], flush=True)
        print(f"{modality} {source}: {len(got)} of {sum(1 for d in doc_source.values() if d == source)} documents "
              f"({first} from the exemplar sample), {time.time() - t0:.0f}s", flush=True)
        return got

    with ThreadPoolExecutor(min(8, max(1, len(jobs)))) as ex:
        for got in ex.map(take, jobs):
            out |= got

    if synth_docs:
        made = ms.synth_images(1.0) if modality == "image" else ms.synth_audio(1.0)
        made = [x for x in made if x[3] == "synth" and "degraded" not in (x[4] or {})]
        matched = 0
        for r, x in zip(synth_docs, made):
            if " | ".join(x[1]) == r["labels"]:
                out[r["doc"]] = encode(x[0])
                matched += 1
        print(f"{modality} synth: {matched} of {len(synth_docs)} drawn again with matching labels", flush=True)
    return out


def pack(modality: str, got: dict[int, bytes], out: Path) -> dict:
    folder = out / "media" / modality
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("p*.bin"):
        old.unlink()
    hosted, k, buf = {}, 0, bytearray()
    for doc in sorted(got):
        data = got[doc]
        if buf and len(buf) + len(data) > PACK_BYTES:
            (folder / f"p{k:03d}.bin").write_bytes(buf)
            k, buf = k + 1, bytearray()
        hosted[doc] = [k, len(buf), len(data)]
        buf += data
    if buf:
        (folder / f"p{k:03d}.bin").write_bytes(buf)
    (folder / "hosted.json").write_text(json.dumps(hosted, separators=(",", ":")))
    print(f"{modality}: {len(hosted)} documents in {k + 1 if buf or hosted else 0} packs", flush=True)
    return hosted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("out")
    ap.add_argument("--sources", default="", help="only these sources (default: every allowed one)")
    ap.add_argument("--modality", default="image,audio")
    args = ap.parse_args()
    catalog, out = Path(args.catalog), Path(args.out)
    only = set(filter(None, args.sources.split(",")))
    for modality in filter(None, args.modality.split(",")):
        sources = set(ALLOWED[modality]) & (only or set(ALLOWED[modality]))
        docs = pq.read_table(catalog / "media" / f"{modality}_docs.parquet").to_pylist()
        pack(modality, fetch(modality, sources, docs), out)
    write_credits(catalog, out)


def write_credits(catalog: Path, out: Path):
    """Credit the sources with at least one document in a pack, whichever runs packed them."""
    credits = {}
    for modality, allowed in ALLOWED.items():
        hosted = out / "media" / modality / "hosted.json"
        if not hosted.exists():
            continue
        shown = set(map(int, json.loads(hosted.read_text())))
        docs = pq.read_table(catalog / "media" / f"{modality}_docs.parquet", columns=["doc", "source"]).to_pylist()
        used = {r["source"] for r in docs if r["doc"] in shown}
        credits[modality] = {s: {"name": n, "license": lic, "credit": c} for s, (n, lic, c) in allowed.items() if s in used}
    (out / "media" / "credits.json").write_text(json.dumps(credits, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
