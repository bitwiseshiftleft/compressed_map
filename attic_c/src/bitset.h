/** @file bitset.h
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Basic bitsets.
 */
#ifndef __LFR_BITSET_H__
#define __LFR_BITSET_H__

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "util.h"

typedef uint64_t bitset_word_t;
#define BITSET_ALIGNMENT (8*sizeof(bitset_word_t))

/**
 * A bitset is just a pointer to an array of words.
 * Structified to discourage lazy dereferencing, so that we can potentially
 * change the impl (eg, from u64 to u8).
 */
typedef struct { bitset_word_t w; } *bitset_t;

/** Allocate memory a bitset whose capacity is at least `nbits` bits.  Set all the bits to zero. */
static inline UNUSED bitset_t bitset_init(size_t nbits) {
    size_t size = BYTES(nbits)+sizeof(bitset_word_t);
    bitset_t ret = (bitset_t)lfr_aligned_alloc(sizeof(bitset_t), size);
    if (ret != NULL) memset(ret,0,size);
    return ret;
}

/** Free the memory used by a bitset */
static inline UNUSED void bitset_destroy(bitset_t bs) {
    free(bs);
}

/** Allocate and return a copy of a bitset */
static inline bitset_t bitset_duplicate(bitset_t bs, size_t nbits) {
    size_t size = BYTES(nbits)+sizeof(bitset_word_t);
    bitset_t ret = (bitset_t)lfr_aligned_alloc(sizeof(bitset_t), size);
    if (ret != NULL) memcpy(ret,bs,size);
    return ret;
}

/** Clear one bit in a bitset */
static inline UNUSED void bitset_set_bit(bitset_t bs, size_t n) {
    bs[n/bitsizeof(bitset_word_t)].w |= (bitset_word_t)1<<(n%bitsizeof(bitset_word_t));
}

/** Toggle one bit in a bitset */
static inline UNUSED void bitset_toggle_bit(bitset_t bs, size_t n) {
    bs[n/bitsizeof(bitset_word_t)].w ^= (bitset_word_t)1<<(n%bitsizeof(bitset_word_t));
}

/** Clear one bit from a bitset */
static inline UNUSED void bitset_clear_bit(bitset_t bs, size_t n) {
    bs[n/bitsizeof(bitset_word_t)].w &= ~((bitset_word_t)1<<(n%bitsizeof(bitset_word_t)));
}

/** Clear all bits from a bitset of size nbits */
static inline UNUSED void bitset_clear_all(bitset_t bs, size_t nbits) {
    memset(bs,0,BYTES(nbits));
}

/** Set all bits in a bitset of size nbits */
static inline UNUSED void bitset_set_all(bitset_t bs, size_t nbits) {
    for (size_t i=0; i<nbits/bitsizeof(bitset_word_t); i++) {
        bs[i].w = ~(bitset_word_t)0;
    }
    if (nbits % bitsizeof(bitset_word_t)) {
        bs[nbits/bitsizeof(bitset_word_t)].w = ((bitset_word_t)1<<(nbits%bitsizeof(bitset_word_t)))-1;
    }
}

/** Toggle up to 8 bits, according to the given byte, starting at index 8*byte_offset */
static inline UNUSED void bitset_toggle_byte(bitset_t bs, size_t byte_offset, uint8_t byte) {
    bs[byte_offset/sizeof(bitset_word_t)].w ^= (bitset_word_t)byte << (8*(byte_offset % sizeof(bitset_word_t)));
}

/** Test if bit `n` in the bitset is set */
static inline UNUSED int bitset_test_bit(const bitset_t bs, size_t n) {
    return 1 & (bs[n/bitsizeof(bitset_word_t)].w >> (n%bitsizeof(bitset_word_t)));
}

/** Count how many bits are set in the first `n` bits of a bitset.  */
static inline UNUSED size_t bitset_popcount(const bitset_t bs, size_t n) {
    size_t ret=0, i=0, width = 8*sizeof(bitset_word_t);
    for (; i<n/width; i++) {
        ret += popcount(bs[i].w);
    }
    if (i*width < n) {
        // last word
        bitset_word_t mask = ((bitset_word_t)1 << (n%width)) - 1;
        ret += popcount(bs[i].w & mask);
    }
    return ret;
}

/** Find the next set bit in the bitset after `start` but before `size`.  Return -1 if there are none. */
static inline UNUSED ssize_t bitset_next_bit(const bitset_t bs, size_t size, size_t start) {
    // TODO optimize
    for (; start<size; start++) {
        if (bitset_test_bit(bs,start)) return start;
    }
    return -1;
}

#endif
