import numpy as np
import pytest

from omamer.alphabets import Alphabet


@pytest.mark.parametrize("n", [21, 13])
def test_sanitiser_keeps_the_21_letters(n):
    # the reduced alphabet merges letters through its lookup table; its sanitiser
    # must keep them, as mkdb sanitises with the 21-letter alphabet
    alphabet = Alphabet(n=n)
    assert alphabet.sanitise_seq("ACDEFGHIKLMNPQRSTVWXY") == "ACDEFGHIKLMNPQRSTVWXY"
    assert alphabet.sanitise_seq("astnqyvrm") == "ASTNQYVRM"
    assert alphabet.sanitise_seq("AB*UZOJ") == "AXXXXXX"


def test_reduced_alphabet_merges_after_sanitising():
    alphabet = Alphabet(n=13)
    lookup = alphabet.DIGITS_AA_LOOKUP
    seq = np.frombuffer(alphabet.sanitise_seq("STNQYVRM").encode(), dtype=np.uint8)
    merged = np.frombuffer(b"AADEFIKL", dtype=np.uint8)
    np.testing.assert_array_equal(lookup[seq], lookup[merged])
