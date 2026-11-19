import numpy as np

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
