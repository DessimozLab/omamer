import numpy as np
import pytest

from omamer.alphabets import Alphabet, get_transform
from omamer.kmer_mask import low_complexity_mask, modality_kmer_mask


def _code(seq, alphabet, k):
    trans = get_transform(k, alphabet.DIGITS_AA)
    return int(sum(int(alphabet.DIGITS_AA_LOOKUP[ord(c)]) * int(t) for c, t in zip(seq, trans)))


def test_low_complexity_mask_counts_distinct_letters():
    alphabet = Alphabet(n=21)
    mask = low_complexity_mask(3, 21, 3)
    assert mask.size == 21**3
    assert mask[_code("DDD", alphabet, 3)]
    assert mask[_code("DVD", alphabet, 3)]
    assert not mask[_code("DVA", alphabet, 3)]
    # 21 one-letter and 21*20*3 two-letter 3-mers
    assert int(mask.sum()) == 21 + 21 * 20 * 3


def test_modality_kmer_mask_per_modality(monkeypatch):
    monkeypatch.delenv("OMAMER_SS_MIN_DISTINCT", raising=False)
    monkeypatch.delenv("OMAMER_SEQ_MIN_DISTINCT", raising=False)
    assert modality_kmer_mask("ss", 3, 21).size == 0
    monkeypatch.setenv("OMAMER_SS_MIN_DISTINCT", "3")
    assert modality_kmer_mask("seq", 3, 21).size == 0
    np.testing.assert_array_equal(modality_kmer_mask("ss", 3, 21), low_complexity_mask(3, 21, 3))
    monkeypatch.setenv("OMAMER_SEQ_MIN_DISTINCT", "2")
    assert int(modality_kmer_mask("seq", 3, 21).sum()) == 21


@pytest.mark.parametrize("k, alphabet_n, min_distinct", [(1, 21, 2), (3, 21, 2), (4, 21, 3), (4, 5, 4)])
def test_low_complexity_mask_matches_sorted_digits(k, alphabet_n, min_distinct):
    codes = np.arange(alphabet_n**k)
    digits = np.stack([(codes // alphabet_n**i) % alphabet_n for i in range(k)], axis=1)
    distinct = np.array([len(set(row)) for row in digits.tolist()])
    np.testing.assert_array_equal(
        low_complexity_mask(k, alphabet_n, min_distinct), distinct < min_distinct
    )
