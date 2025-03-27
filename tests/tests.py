import numpy as np
import numba

<<<<<<< Updated upstream
from omamer.index import popcount, select1_in_word
=======
from omamer.index import ctz, naive_ctz, popcount, select1_in_word
from omamer.index import to_elias_fano, from_elias_fano, from_elias_fano_correct
>>>>>>> Stashed changes


def popcount_naive(x):
    """
    Naive implementation of popcount, i.e. counting
    the number of 1s in the bitwise representation of x
    """
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

def test_popcount():
    n = 500_000
    data = np.random.randint(0, np.iinfo(np.uint32).max,
                             size=n, dtype=np.uint32)

    for i in range(n):
        x = data[i]
        expected = popcount_naive(x)
        actual = popcount(x)
        assert expected == actual

@numba.njit
def select1_in_word_naive(word, rank):
    count = 0
    for bit in range(32):
        if (word >> bit) & 1:
            if count == rank:
                return bit
            count += 1
    return -1

<<<<<<< Updated upstream
# ✅ Unit test
=======

>>>>>>> Stashed changes
def test_select1_in_word():
    n = 100_000
    for _ in range(n):
        word = np.random.randint(0, 2**32, dtype=np.uint32)
        pop = bin(word).count("1")
        for r in range(pop):
            expected = select1_in_word_naive(word, r)
            actual = select1_in_word(word, r)
            assert expected == actual, f"FAIL: word={bin(word)}, rank={r}, expected={expected}, got={actual}"

        # Check rank overflow case
        assert select1_in_word(word, pop) == -1, f"FAIL: rank={pop} should be invalid"
<<<<<<< Updated upstream
=======

def test_ctz():
    seed = 42
    num_tests = 10000
    np.random.seed(seed)

    for _ in range(num_tests):
        v = np.random.randint(0, 2**32, dtype=np.uint32)
        expected = naive_ctz(v)
        actual = ctz(v)
        assert expected == actual, f"Error for value {v}"


def test_elias_fano(num_tests=1000, max_len=200, max_val=10**6):
    for _ in range(num_tests):
        length = np.random.randint(1, max_len + 1)
        values = np.sort(np.random.choice(
            np.arange(max_val, dtype=np.uint32), size=length, replace=False))

        ef = to_elias_fano(values)

        # assert np.array_equal(from_elias_fano_correct(ef.l,
        #     ef.lower_packed,
        #     ef.upper_packed,
        #     ef.n), values)

        decoded = from_elias_fano(
            ef.l,
            ef.lower_packed,
            ef.upper_packed,
            ef.n
        )

        assert np.array_equal(decoded, values), f"FAILED:\nOriginal: {values}\nDecoded: {decoded}"
>>>>>>> Stashed changes
