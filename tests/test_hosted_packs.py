"""The hosted pack builder's pure helpers: the packed vocabulary mask and the population split."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "browser-runtime"))
sys.path.insert(0, str(ROOT / "drowse"))

from build_hosted_packs import mask_words, split_population, unpack_mask_words  # noqa: E402


def test_mask_words_set_bit_v_of_word_v_over_32() -> None:
    keep = torch.zeros(70, dtype=torch.bool)
    keep[[0, 31, 32, 69]] = True
    words = mask_words(keep)
    assert words.dtype == torch.uint32 and words.shape == (3,)
    assert [int(w) for w in words.to(torch.int64)] == [1 | (1 << 31), 1, 1 << 5]
    assert torch.equal(unpack_mask_words(words, 70), keep)


def test_split_population_keeps_vocabulary_statistics_apart() -> None:
    names = [
        "R.writes.rho", "R.writes.lse", "R.writes.mean", "R.writes.std", "S.fires.mean",
        "S.fires+.norm", "R.heads.ov.std", "heads.norm_derivative.query", "neurons.write_norms",
    ]
    reads, contrast = split_population({name: torch.zeros(1) for name in names})
    assert sorted(contrast) == ["R.heads.ov.std", "R.writes.mean", "R.writes.std", "S.fires.mean"]
    assert sorted(reads) == [
        "R.writes.lse", "R.writes.rho", "S.fires+.norm", "heads.norm_derivative.query", "neurons.write_norms",
    ]
