from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).with_name("prepare_hosted_release_corpus.py")
SPEC = importlib.util.spec_from_file_location("prepare_hosted_release_corpus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corpus)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ReleaseCorpusToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.json"
        rows = [
            f"Document row {index} contains enough stable text for deterministic release preparation."
            for index in range(8)
        ]
        self.source.write_bytes((json.dumps(rows, separators=(",", ":")) + "\n").encode())
        self.evidence = self.root / "LICENSE.txt"
        self.evidence.write_text("Fixture corpus is released under CC0-1.0.\n")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self.make_manifest()
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_manifest(self) -> dict:
        return {
            "$schema": "./release-corpus-source.schema.json",
            "schemaVersion": 2,
            "status": "approved",
            "blockers": [],
            "source": {
                "repository": "fixture/corpus",
                "revision": "a" * 40,
                "configuration": "plain",
                "split": "train",
                "path": "data/train.json",
                "url": f"https://example.invalid/fixture/corpus/resolve/{'a' * 40}/data/train.json",
                "format": "json",
                "textField": "text",
                "sha256": sha256(self.source.read_bytes()),
                "bytes": self.source.stat().st_size,
                "rows": 8,
            },
            "license": {
                "declaredIdentifiers": ["CC0-1.0"],
                "evidence": {
                    "url": f"https://example.invalid/fixture/corpus/raw/{'a' * 40}/LICENSE.txt",
                    "sha256": sha256(self.evidence.read_bytes()),
                    "bytes": self.evidence.stat().st_size,
                },
                "attribution": "Release corpus fixture.",
                "redistributionNotice": "CC0-1.0; attribution retained for provenance.",
                "reviewStatus": "approved",
                "reviewedBy": "Drowse release fixture",
                "reviewedAt": "2026-08-29T00:00:00Z",
            },
            "selection": {
                "algorithm": "drowse-release-corpus-v1",
                "minimumFragmentCharacters": 10,
                "minimumDocumentCharacters": 40,
                "maximumDocumentCharacters": 120,
                "maximumSourceBytes": 4096,
                "maximumSourceRows": 16,
                "saeDocuments": 3,
            },
        }

    def write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")

    def prepare(self, output_name: str = "output") -> dict:
        return corpus.prepare(
            self.manifest_path,
            self.source,
            self.evidence,
            self.root / output_name,
        )

    def test_approved_inputs_generate_deterministic_bounded_outputs(self) -> None:
        first = self.prepare("first")
        self.prepare("second")
        # Only the SAE corpus: the hosted lenses are native fits on their own pinned corpus.
        self.assertEqual(sorted(path.name for path in (self.root / "first").iterdir()),
                         ["release-corpus.json", "sae-corpus.json"])
        for name in ("sae-corpus.json", "release-corpus.json"):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())
        self.assertEqual(first["$schema"], "https://polythetic.ai/schemas/release-corpus-v2.json")
        self.assertEqual(first["schemaVersion"], 2)
        self.assertEqual(list(first["outputs"]), ["sae"])
        self.assertEqual(first["outputs"]["sae"]["documents"], 3)
        output = first["outputs"]["sae"]
        payload = (self.root / "first" / output["path"]).read_bytes()
        self.assertEqual(output["sha256"], sha256(payload))
        self.assertLessEqual(
            max(map(len, json.loads(payload))),
            self.manifest["selection"]["maximumDocumentCharacters"],
        )

    def test_retired_lens_fields_are_refused(self) -> None:
        for key, value in (("wordSeeds", {}), ("tokenizers", [])):
            manifest = json.loads(json.dumps(self.manifest))
            manifest[key] = value
            self.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(corpus.ReleaseCorpusError, "fields must be exactly"):
                self.prepare()
        self.manifest["selection"]["jlensDocuments"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "fields must be exactly"):
            self.prepare()

    def test_blocked_candidate_fails_before_reading_missing_inputs(self) -> None:
        candidates = [
            "wikitext-103-raw-test.candidate.json",
            "fineweb-sample-10bt-train.candidate.json",
        ]
        for name in candidates:
            candidate = REPOSITORY_ROOT / "browser-runtime/release-corpus" / name
            value = json.loads(candidate.read_text())
            corpus._validate_manifest(value)
            output = self.root / f"blocked-output-{name}"
            with self.assertRaisesRegex(corpus.ReleaseCorpusError, "release corpus source is blocked"):
                corpus.prepare(
                    candidate,
                    self.root / "missing-source",
                    self.root / "missing-evidence",
                    output,
                )
            self.assertFalse(output.exists())

    def test_source_digest_mismatch_fails_without_output(self) -> None:
        self.source.write_text("corrupt\n")
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "source snapshot size mismatch"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_license_evidence_mismatch_fails_without_output(self) -> None:
        self.evidence.write_text("changed license evidence\n")
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "license evidence size mismatch"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_unknown_manifest_field_is_rejected(self) -> None:
        self.manifest["assumedLicense"] = True
        self.write_manifest()
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "fields must be exactly"):
            self.prepare()

    def test_duplicate_manifest_field_is_rejected(self) -> None:
        self.manifest_path.write_text('{"schemaVersion":2,"schemaVersion":2}\n')
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "repeats the JSON field"):
            self.prepare()

    def test_approved_manifest_requires_review_identity(self) -> None:
        self.manifest["license"]["reviewedBy"] = None
        self.write_manifest()
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "named license reviewer"):
            self.prepare()

    def test_insufficient_documents_fail_without_partial_output(self) -> None:
        self.manifest["selection"]["saeDocuments"] = 10
        self.write_manifest()
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "bounded documents"):
            self.prepare()
        self.assertFalse((self.root / "output").exists())

    def test_parquet_rows_stream_without_materializing_the_source(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            f"Parquet row {index} contains enough stable text for deterministic release preparation."
            for index in range(8)
        ]
        parquet_path = self.root / "source.parquet"
        pq.write_table(pa.table({"text": rows}), parquet_path, row_group_size=2)
        self.source = parquet_path
        self.manifest["source"].update({
            "path": "data/train.parquet",
            "url": f"https://example.invalid/fixture/corpus/resolve/{'a' * 40}/data/train.parquet",
            "format": "parquet",
            "sha256": sha256(parquet_path.read_bytes()),
            "bytes": parquet_path.stat().st_size,
            "rows": len(rows),
        })
        self.write_manifest()
        result = self.prepare()
        self.assertEqual(result["outputs"]["sae"]["documents"], 3)

    def test_document_limit_stops_consuming_streamed_rows(self) -> None:
        def rows():
            yield "A first bounded document with enough text to finish immediately."
            raise AssertionError("the document packer consumed rows beyond its requested limit")

        result = corpus._pack_documents(
            rows(),
            minimum_fragment_characters=10,
            minimum_document_characters=20,
            maximum_document_characters=40,
            limit=1,
        )
        self.assertEqual(len(result), 1)

    def test_output_directory_is_never_overwritten(self) -> None:
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(corpus.ReleaseCorpusError, "already exists"):
            self.prepare()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_control_characters_are_rejected(self) -> None:
        self.assertRaisesRegex(
            corpus.ReleaseCorpusError,
            "control character",
            corpus._normalize_text,
            "visible\x00hidden",
        )

def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
