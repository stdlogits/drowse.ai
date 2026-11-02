from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


ALGORITHM = "drowse-release-corpus-v1"
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_ROWS = 2_000_000
MAX_OUTPUT_DOCUMENTS = 10_000
MAX_DOCUMENT_CHARACTERS = 32_768
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ReleaseCorpusError(ValueError):
    pass


def prepare(
    manifest_path: Path,
    source_path: Path,
    license_evidence_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """The SAE release corpus (``sae-corpus.json``) and its manifest from an approved source.
    Hosted R and S lenses are drowse's native fits on their own pinned corpus
    (``browser-runtime/release-corpus/native-lens-corpus.json``), so this corpus feeds only the
    offline SAE builder."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json(manifest_bytes, "release corpus source manifest")
    _validate_manifest(manifest)
    if manifest["status"] != "approved":
        reasons = "; ".join(manifest["blockers"])
        raise ReleaseCorpusError(f"release corpus source is blocked: {reasons}")

    if output_directory.exists():
        raise ReleaseCorpusError(f"output directory already exists: {output_directory}")
    _verify_file(source_path, manifest["source"], "source snapshot")
    _verify_file(license_evidence_path, manifest["license"]["evidence"], "license evidence")

    selection = manifest["selection"]
    rows = _read_source_rows(source_path, manifest["source"], selection)
    documents = _pack_documents(
        rows,
        minimum_fragment_characters=selection["minimumFragmentCharacters"],
        minimum_document_characters=selection["minimumDocumentCharacters"],
        maximum_document_characters=selection["maximumDocumentCharacters"],
        limit=selection["saeDocuments"],
    )
    if len(documents) < selection["saeDocuments"]:
        raise ReleaseCorpusError(
            f"source produced {len(documents)} bounded documents; {selection['saeDocuments']} are required"
        )

    output_payloads = {"sae-corpus.json": _canonical_json(documents)}
    output_manifest = {
        "$schema": "https://polythetic.ai/schemas/release-corpus-v2.json",
        "schemaVersion": 2,
        "algorithm": ALGORITHM,
        "generatorSha256": _sha256(Path(__file__).read_bytes()),
        "sourceManifestSha256": _sha256(manifest_bytes),
        "source": manifest["source"],
        "license": manifest["license"],
        "selection": selection,
        "outputs": {
            "sae": _output_record("sae-corpus.json", output_payloads["sae-corpus.json"], documents),
        },
    }
    output_payloads["release-corpus.json"] = _pretty_json(output_manifest)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".drowse-release-corpus-", dir=output_directory.parent))
    try:
        for name, payload in output_payloads.items():
            path = stage / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(stage, output_directory)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output_manifest


def _validate_manifest(value: Any) -> None:
    _exact_keys(
        value,
        {
            "$schema",
            "schemaVersion",
            "status",
            "blockers",
            "source",
            "license",
            "selection",
        },
        "release corpus source manifest",
    )
    if value["schemaVersion"] != 2:
        raise ReleaseCorpusError("release corpus source manifest schemaVersion must be 2")
    if not _nonempty_text(value["$schema"]):
        raise ReleaseCorpusError("release corpus source manifest $schema must be a non-empty string")
    if value["status"] not in {"approved", "blocked"}:
        raise ReleaseCorpusError("release corpus source manifest status must be approved or blocked")
    blockers = value["blockers"]
    if (
        not isinstance(blockers, list)
        or any(not _nonempty_text(item) for item in blockers)
        or len(set(blockers)) != len(blockers)
    ):
        raise ReleaseCorpusError("release corpus blockers must be unique non-empty strings")
    if value["status"] == "approved" and blockers:
        raise ReleaseCorpusError("an approved release corpus source cannot retain blockers")
    if value["status"] == "blocked" and not blockers:
        raise ReleaseCorpusError("a blocked release corpus source must explain its blockers")
    _validate_source(value["source"])
    _validate_license(value["license"], approved=value["status"] == "approved")
    _validate_selection(value["selection"])


def _validate_source(value: Any) -> None:
    _exact_keys(
        value,
        {
            "repository",
            "revision",
            "configuration",
            "split",
            "path",
            "url",
            "format",
            "textField",
            "sha256",
            "bytes",
            "rows",
        },
        "source",
    )
    if not isinstance(value["repository"], str) or not REPOSITORY.fullmatch(value["repository"]):
        raise ReleaseCorpusError("source repository must be an exact owner/name coordinate")
    if not isinstance(value["revision"], str) or not REVISION.fullmatch(value["revision"]):
        raise ReleaseCorpusError("source revision must be a lowercase 40-character commit")
    for key in ("configuration", "split", "textField"):
        if not _nonempty_text(value[key]):
            raise ReleaseCorpusError(f"source {key} must be a non-empty string")
    _relative_path(value["path"], "source path")
    parsed = urlparse(value["url"] if isinstance(value["url"], str) else "")
    if parsed.scheme != "https" or not parsed.netloc or value["revision"] not in value["url"]:
        raise ReleaseCorpusError("source URL must be HTTPS and contain the exact source revision")
    if value["format"] not in {"json", "jsonl", "parquet"}:
        raise ReleaseCorpusError("source format must be json, jsonl, or parquet")
    _digest(value["sha256"], "source sha256")
    _positive_integer(value["bytes"], "source bytes")
    _positive_integer(value["rows"], "source rows")


def _validate_license(value: Any, *, approved: bool) -> None:
    _exact_keys(
        value,
        {
            "declaredIdentifiers",
            "evidence",
            "attribution",
            "redistributionNotice",
            "reviewStatus",
            "reviewedBy",
            "reviewedAt",
        },
        "license",
    )
    identifiers = value["declaredIdentifiers"]
    if (
        not isinstance(identifiers, list)
        or not identifiers
        or any(not _nonempty_text(item) for item in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ReleaseCorpusError("license identifiers must be unique non-empty strings")
    _validate_evidence(value["evidence"])
    for key in ("attribution", "redistributionNotice"):
        if not _nonempty_text(value[key]):
            raise ReleaseCorpusError(f"license {key} must be a non-empty string")
    if value["reviewStatus"] not in {"approved", "blocked"}:
        raise ReleaseCorpusError("license reviewStatus must be approved or blocked")
    if approved:
        if value["reviewStatus"] != "approved":
            raise ReleaseCorpusError("approved corpus source requires approved license review")
        if not _nonempty_text(value["reviewedBy"]):
            raise ReleaseCorpusError("approved corpus source requires a named license reviewer")
        if not _rfc3339(value["reviewedAt"]):
            raise ReleaseCorpusError("approved corpus source requires an RFC3339 license review time")
    elif value["reviewStatus"] != "blocked":
        raise ReleaseCorpusError("blocked corpus source must keep its license review blocked")
    elif value["reviewedBy"] is not None or value["reviewedAt"] is not None:
        raise ReleaseCorpusError("blocked corpus source cannot claim a completed license review")


def _validate_evidence(value: Any) -> None:
    _exact_keys(value, {"url", "sha256", "bytes"}, "license evidence")
    parsed = urlparse(value["url"] if isinstance(value["url"], str) else "")
    if parsed.scheme != "https" or not parsed.netloc:
        raise ReleaseCorpusError("license evidence URL must be HTTPS")
    _digest(value["sha256"], "license evidence sha256")
    _positive_integer(value["bytes"], "license evidence bytes")


def _validate_selection(value: Any) -> None:
    _exact_keys(
        value,
        {
            "algorithm",
            "minimumFragmentCharacters",
            "minimumDocumentCharacters",
            "maximumDocumentCharacters",
            "maximumSourceBytes",
            "maximumSourceRows",
            "saeDocuments",
        },
        "selection",
    )
    if value["algorithm"] != ALGORITHM:
        raise ReleaseCorpusError(f"selection algorithm must be {ALGORITHM}")
    for key in (
        "minimumFragmentCharacters",
        "minimumDocumentCharacters",
        "maximumDocumentCharacters",
        "maximumSourceBytes",
        "maximumSourceRows",
        "saeDocuments",
    ):
        _positive_integer(value[key], f"selection {key}")
    if value["minimumFragmentCharacters"] > value["minimumDocumentCharacters"]:
        raise ReleaseCorpusError("minimum fragment characters cannot exceed the document minimum")
    if value["minimumDocumentCharacters"] > value["maximumDocumentCharacters"]:
        raise ReleaseCorpusError("document minimum cannot exceed its maximum")
    if value["maximumDocumentCharacters"] > MAX_DOCUMENT_CHARACTERS:
        raise ReleaseCorpusError("document maximum exceeds the release-tool bound")
    if value["maximumSourceBytes"] > MAX_SOURCE_BYTES:
        raise ReleaseCorpusError("source byte limit exceeds the release-tool bound")
    if value["maximumSourceRows"] > MAX_SOURCE_ROWS:
        raise ReleaseCorpusError("source row limit exceeds the release-tool bound")
    if value["saeDocuments"] > MAX_OUTPUT_DOCUMENTS:
        raise ReleaseCorpusError("document count exceeds the release-tool bound")


def _verify_file(path: Path, record: dict[str, Any], label: str) -> None:
    info = path.stat()
    if not path.is_file():
        raise ReleaseCorpusError(f"{label} is not a regular file: {path}")
    if info.st_size != record["bytes"]:
        raise ReleaseCorpusError(
            f"{label} size mismatch: expected {record['bytes']}, received {info.st_size}"
        )
    digest = _sha256_file(path)
    if digest != record["sha256"]:
        raise ReleaseCorpusError(
            f"{label} digest mismatch: expected {record['sha256']}, received {digest}"
        )


def _read_source_rows(
    path: Path, source: dict[str, Any], selection: dict[str, Any]
) -> Iterator[str]:
    if source["bytes"] > selection["maximumSourceBytes"]:
        raise ReleaseCorpusError("source snapshot exceeds maximumSourceBytes")
    if source["rows"] > selection["maximumSourceRows"]:
        raise ReleaseCorpusError("source snapshot exceeds maximumSourceRows")
    if source["format"] == "json":
        value = _load_json(path.read_bytes(), "source JSON")
        if not isinstance(value, list):
            raise ReleaseCorpusError("source JSON must be an array")
        rows = [_source_text(item, source["textField"], index) for index, item in enumerate(value)]
    elif source["format"] == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    raise ReleaseCorpusError(f"source JSONL row {index} is blank")
                rows.append(
                    _source_text(
                        _load_json(line.encode(), f"source JSONL row {index}"),
                        source["textField"],
                        index,
                    )
                )
    else:
        return _parquet_rows(path, source["textField"], source["rows"])
    if len(rows) != source["rows"]:
        raise ReleaseCorpusError(
            f"source row count mismatch: expected {source['rows']}, received {len(rows)}"
        )
    return iter(rows)


def _parquet_rows(path: Path, text_field: str, expected_rows: int) -> Iterator[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ReleaseCorpusError(
            "Parquet corpus preparation requires pyarrow in the release environment"
        ) from error
    source = parquet.ParquetFile(path)
    if source.metadata.num_rows != expected_rows:
        raise ReleaseCorpusError(
            f"source row count mismatch: expected {expected_rows}, received {source.metadata.num_rows}"
        )

    def rows() -> Iterator[str]:
        index = 0
        for batch in source.iter_batches(
            batch_size=4096,
            columns=[text_field],
            use_threads=False,
        ):
            for value in batch.column(0).to_pylist():
                yield _source_text(value, text_field, index)
                index += 1

    return rows()


def _source_text(value: Any, text_field: str, index: int) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or set(value) != {text_field} or not isinstance(value[text_field], str):
        raise ReleaseCorpusError(
            f"source row {index} must be a string or an object containing only {text_field!r}"
        )
    return value[text_field]


def _pack_documents(
    rows: Iterable[str],
    *,
    minimum_fragment_characters: int,
    minimum_document_characters: int,
    maximum_document_characters: int,
    limit: int,
) -> list[str]:
    documents: list[str] = []
    current = ""
    for row in rows:
        normalized = _normalize_text(row)
        if len(normalized) < minimum_fragment_characters:
            continue
        for fragment in _split_fragment(normalized, maximum_document_characters):
            candidate = f"{current} {fragment}" if current else fragment
            if len(candidate) <= maximum_document_characters:
                current = candidate
                continue
            if len(current) >= minimum_document_characters:
                documents.append(current)
                if len(documents) == limit:
                    return documents
                current = fragment
            else:
                current = candidate[:maximum_document_characters].rstrip()
                documents.append(current)
                if len(documents) == limit:
                    return documents
                current = candidate[maximum_document_characters:].lstrip()
    if len(current) >= minimum_document_characters and len(documents) < limit:
        documents.append(current)
    return documents


def _split_fragment(value: str, maximum: int) -> list[str]:
    parts = []
    remainder = value
    while len(remainder) > maximum:
        split = remainder.rfind(" ", 0, maximum + 1)
        if split <= 0:
            split = maximum
        parts.append(remainder[:split].rstrip())
        remainder = remainder[split:].lstrip()
    if remainder:
        parts.append(remainder)
    return parts


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(character) == "Cc" and not character.isspace() for character in normalized):
        raise ReleaseCorpusError("source text contains a non-whitespace control character")
    return " ".join(normalized.split())


def _output_record(path: str, payload: bytes, documents: list[str]) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "documents": len(documents),
        "characters": sum(len(document) for document in documents),
    }


def _load_json(value: bytes, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ReleaseCorpusError(f"{label} repeats the JSON field {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ReleaseCorpusError(f"{label} contains the non-finite JSON number {constant}")

    try:
        return json.loads(value, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseCorpusError(f"{label} is not valid UTF-8 JSON") from error


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _pretty_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReleaseCorpusError(f"{label} fields must be exactly {sorted(expected)}; received {actual}")


def _relative_path(value: Any, label: str) -> None:
    if not _nonempty_text(value) or "\\" in value or "\x00" in value:
        raise ReleaseCorpusError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseCorpusError(f"{label} must be a canonical relative POSIX path")


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ReleaseCorpusError(f"{label} must be a lowercase SHA-256 digest")


def _positive_integer(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReleaseCorpusError(f"{label} must be a positive integer")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _rfc3339(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic, provenance-bound hosted SAE corpora."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("license_evidence", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    result = prepare(
        arguments.manifest.resolve(),
        arguments.source.resolve(),
        arguments.license_evidence.resolve(),
        arguments.output_directory.resolve(),
    )
    sys.stdout.write(f"{json.dumps(result, sort_keys=True)}\n")


if __name__ == "__main__":
    main()
