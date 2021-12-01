/** @file util.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 * Utility functions and macros.
 */

#ifndef __LFR_UTIL_H__
#define __LFR_UTIL_H__

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> /* for memcpy */
#include <sys/types.h> /* for ssize_t */

/* Builtin checking */
#ifndef LFR_USE_BUILTINS
#define LFR_USE_BUILTINS 1
#endif
#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif

/* Unused function */
#ifndef _MSC_VER
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

/* Define ssize_t */
#ifdef _MSC_VER
    typedef long long ssize_t;  /* win-64*/
    #define SSIZE_MIN  LLONG_MIN
    #define SSIZE_MAX  LLONG_MAX
#endif

/* Make visible in object file */
#define API_VIS __attribute__((visibility("default")))

#define BYTES(bits) (((bits)+7)/8)
#define bitsizeof(x) (8*sizeof(x))

/** Return ceil(sz/align) */
static inline UNUSED size_t div_round_up(size_t sz, size_t align) {
    return (sz+align-1) / align;
}

/** Return align*ceil(sz/align) */
static inline UNUSED size_t round_up(size_t sz, size_t align) {
    return div_round_up(sz,align) * align;
}

/** Count number of set bits */ 
static inline UNUSED int popcount(uint64_t x) {
#if LFR_USE_BUILTINS && __has_builtin(__builtin_popcountll)
    return __builtin_popcountll(x);
#else
    // From Wikipedia
    const uint64_t m1  = 0x5555555555555555, m2  = 0x3333333333333333;
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f, h01 = 0x0101010101010101;
    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    return (x * h01) >> 56;
#endif
}

/** Count parity of number of set bits: 1 for odd and 0 for even */ 
static inline UNUSED int parity(uint64_t x) {
#if LFR_USE_BUILTINS && __has_builtin(__builtin_parityll)
    return __builtin_parityll(x);
#else
    x ^= x>>32;
    x ^= x>>16;
    x ^= x>>8;
    x ^= x>>4;
    return (0x6996>>(x&0xF)) & 1;
#endif
}

/** Count trailing zeros */
static inline UNUSED int ctz(uint64_t x) {
#if LFR_USE_BUILTINS && __has_builtin(__builtin_ctzll)
    if (x==0) return 8*sizeof(x);
    return __builtin_ctzll(x);
#else
    return popcount((x-1) &~ x);
#endif
}

/** Return the position of the high bit of x.  Returns -1 if x==0, since it has no high bit. */
static inline UNUSED int high_bit(uint64_t x) {
    if (x==0) return -1;
#if LFR_USE_BUILTINS && __has_builtin(__builtin_clzll)
    return 8*sizeof(x) - 1 - __builtin_clzll(x);
#else
    // Based on Wikipedia implementation
    int r, q;
    r = (x > 0xFFFFFFFF) << 5; x >>= r;
    q = (x >     0xFFFF) << 4; x >>= q; r |= q;
    q = (x >       0xFF) << 3; x >>= q; r |= q;
    q = (x >        0xF) << 2; x >>= q; r |= q;
    q = (x >        0x3) << 1; x >>= q; r |= q;
    return r | (x>>1);
#endif
}

/** Swap memory contents between two pointers.  The memory referred to must not alias. */
static inline UNUSED void memswap(uint8_t *__restrict__ x, uint8_t *__restrict__ y, size_t length) {
    for (unsigned i=0; i<length; i++) {
        uint8_t tmp = x[i];
        x[i] = y[i];
        y[i] = tmp;
    }
}

/** Allocate an aligned region of memory, preferring malloc since it's faster.  */
static inline UNUSED void* lfr_aligned_alloc(size_t alignment, size_t size) {
    if (alignment <= sizeof(void*)) {
        return malloc(round_up(size,alignment));
    } else {
        return aligned_alloc(alignment, round_up(size,alignment));
    }
}

/** Read  and return a little-endian word of `len`<8 bytes at offset `le` */
static inline UNUSED uint64_t le2ui(const uint8_t *le, unsigned len) {
    uint64_t ret = 0;
    assert(len <= sizeof(ret));
    for (unsigned i=0; i<len; i++) {
        ret |= (uint64_t)le[i] << (8*i);
    }
    return ret;
}

/** Write a little-endian word of `len` bytes at offset `le`
 * Return 0 on success, or -1 if it was too big to fit there.
 */
static inline UNUSED int ui2le(uint8_t *le, unsigned len, uint64_t ui) {
    for (unsigned i=0; i<len; i++) {
        le[i] = ui;
        ui >>= 8;
    }
    
    return ui ? -1 : 0;
}

/** Calculate a*b + c unless it would overflow an ssize_t,
 * in which case return -1.  Intended to be used with all
 * positive numbers.
 */
static ssize_t UNUSED safe_mul_add(size_t a, size_t b, size_t c) {
    if ((ssize_t)a < 0 || (ssize_t)b < 0 || (ssize_t)c < 0) return -1;
    if (a==0 || b==0) return c;
    ssize_t prod = (ssize_t)a * b;
    if (prod < 0 || prod/(ssize_t)a != (ssize_t)b) return -1;
    prod += c;
    if (prod < (ssize_t)c) return -1;
    return prod;
}

/** Hash utility: rotate left */
static inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

/** Hash utility: murmur3 fmix */
static inline uint64_t fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}

/** Hash utility: murmur3 hash result */
typedef struct { uint64_t low64, high64; } hash_result_t;

/** Hash utility: Murmur3 by Austin Appleby, but with its seed extended to 64 bits.
 * Based on public-domain code by Peter Scott: https://github.com/PeterScott/murmur3
 */
static hash_result_t UNUSED murmur3_x64_128_extended_seed (
    const uint8_t *data, size_t len, uint64_t seed
) {
    uint64_t len_orig = len;
    uint64_t h1 = seed, h2 = seed;
    uint64_t c1 = 0x87c37b91114253d5ull, c2 = 0x4cf5ad432745937full;

    // Process blocks of 16 bytes
#if __clang__
    #pragma clang loop vectorize(disable) // small trip count, not worth it
#endif
    for(; len >= 16; len -= 16, data += 16) {
        h1 ^= rotl64(le2ui(data  ,8)*c1,31)*c2;
        h2 ^= rotl64(le2ui(data+8,8)*c2,33)*c1;
        h1 = (rotl64(h1,27)+h2)*5+0x52dce729;
        h2 = (rotl64(h2,31)+h1)*5+0x38495ab5;
    }

    // Process the rest
    uint8_t rest[16] = {0};
    memcpy(rest,data,len);
    h1 ^= rotl64(le2ui( rest,   8)*c1, 31)*c2 ^ len_orig;
    h2 ^= rotl64(le2ui(&rest[8],8)*c2, 33)*c1 ^ len_orig;

    // Finalize
    h1 += h2; h2 += h1;
    h1 = fmix64(h1); h2 = fmix64(h2);
    h1 += h2; h2 += h1;

    hash_result_t ret = { h1, h2 };
    return ret;
}

#endif // __LFR_UTIL_H__
