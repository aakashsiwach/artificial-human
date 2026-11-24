"""Opt-in adapters for independently authored evaluation datasets.

Adapters read a dataset that the operator has downloaded locally. They never
download data, never call a remote model, and report only aggregates: no
conversation text, question text, or participant identifiers. Provenance,
licences and download instructions are in ``docs/benchmarks/EXTERNAL_DATASETS.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


class ExternalDatasetError(ValueError):
    """An external dataset is missing, malformed, or fails validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_manifest(paths: Iterable[Path]) -> str:
    """Hash of sorted ``name<TAB>sha256`` lines, for datasets made of many files."""
    lines = sorted(f"{path.name}\t{sha256_file(path)}" for path in paths)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
