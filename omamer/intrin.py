import cppyy
import cppyy.numba_ext


cppyy.cppdef(r"""
#include <stdint.h>

extern "C" uint32_t ctz32(uint32_t x) {
    return __builtin_ctz(x);
}

extern "C"
uint32_t extract_lower_bits(uint32_t word, uint32_t next_word,
                            uint32_t offset, uint32_t bit_width) {
    uint32_t value = word >> offset;
    if (offset + bit_width > 32) {
        value |= next_word << (32 - offset);
    }
    return value & ((1U << bit_width) - 1);
}
""")

ctz_cpp = cppyy.gbl.ctz32
