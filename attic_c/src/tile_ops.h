/**
 * @file tile_ops.h
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Tile matrix operations.  This library implements operations on 8x8 matrices,
 * represented by 64-bit integers.  The matrices are stored in column-major order,
 * i.e. with column 0 as bits 0..7 of the integer.
 */
 
#ifndef __TILE_OPS_H__
#define __TILE_OPS_H__

#include <stdint.h>
#include "util.h"

#define TILE_SIZE 8
typedef uint64_t tile_t; /** 8x8 matrix tile */
typedef uint8_t tile_edge_t; /** 1x8 edge of a tile matrix */
#define TILES_SPANNING(n) (((n)+TILE_SIZE-1)/TILE_SIZE)

/** Return a tile_edge_t with all bits set */
static inline tile_edge_t tile_edge_full() { return 0xFF; }

/** Return a tile_edge_t with no bits set */
static inline tile_edge_t tile_edge_zero() { return 0x00; }

/** Return the identity tile */
static inline tile_t tile_identity() { return 0x8040201008040201ull; }

/** Return all-zero tile */
static inline tile_t tile_zero() { return 0; }

/** Return all-ones tile */
static inline tile_t tile_full() { return 0xFFFFFFFFFFFFFFFFull; }

/** Make a random tile using random(). */
static inline tile_t tile_random() {
    return random() ^ (tile_t)random()<<20 ^ (tile_t)random()<<40;
}

/** If really==1, return a matrix with only one bit set, at [row,column].
 *  If really==0, return the all-zeros tile.
 */
static inline tile_t tile_single_bit(int row, int col, int really) {
    return (tile_t)really << (col*8+row);
}

/** Return the transpose of a tile */
static inline tile_t tile_transpose(tile_t a) {
    a = (a & 0xF0F0F0F00F0F0F0Full)
      | ((a<<28) & 0x0F0F0F0F00000000ull)
      | ((a&0x0F0F0F0F00000000ull)>>28); // transpose blocks of size 4

    a = (a & 0xCCCC3333CCCC3333ull)
      | ((a<<14) & 0x3333000033330000ull)
      | ((a & 0x3333000033330000ull)>>14); // size 2

    a = (a & 0xAA55AA55AA55AA55ull)
      | ((a<<7) & 0x5500550055005500ull)
      | ((a & 0x5500550055005500ull)>>7); // size 1
    
    return a;
}

/** Broadcast the given row (numbered 0..7) to cover the whole tile. */
static inline tile_t tile_broadcast_row(tile_t a, int row) {
    a = (a>>row) & 0x0101010101010101ull;
    return (a<<8)-a;
}

/** Broadcast the given column (numbered 0..7) to cover the whole tile. */
static inline tile_t tile_broadcast_col(tile_t a, int col) {
    a = (a>>(8*col)) & 0xFF;
    return a * 0x0101010101010101ull;
}

/** Return the bit at [row,col] in the tile. */
static inline int tile_get_bit(tile_t a, int row, int col) {
    return 1 & (a>> (col*8+row));
}

/** Return a tile, each of whose columns is `edge`. */
static inline tile_t tile_broadcast_edge_as_col(tile_edge_t edge) {
    return edge * 0x0101010101010101ull;
}

/** Multiply two 8x8 matrices */
static inline tile_t tile_mul(tile_t a, tile_t b) {
    tile_t ret = tile_zero();
    for (unsigned i=0; i<TILE_SIZE; i++) {
        ret ^= tile_broadcast_col(a, i) & tile_broadcast_row(b, i);
    }
    return ret;
}

/** Return a mask of all bits in a given row */
static inline tile_t tile_row_mask(int row) {
    return 0x0101010101010101ull << row;
}

/** Return a mask of all bits in a given column */
static inline tile_t tile_col_mask(int col) {
    return 0xFFull << (8*col);
}

/** Return a mask of all bits in n nows: [row, row+n-1] */
static inline tile_t tile_row_bulk_mask(int row, int n) {
    return tile_broadcast_edge_as_col((1<<(row+n))-(1<<row));
}

/** Return a mask of all bits in n column: [col, col+n-1] */
static inline tile_t tile_col_bulk_mask(int col, int n) {
    tile_t m = (tile_t)1<<(8*col);
    return (n >= 8) ? -m : ((m<<(8*n)) - m);
}

/** Return tile a, with b[colb] copied to a[cola] */
static inline tile_t tile_copy_col(tile_t a, tile_t b, int cola, int colb) {
    tile_t ma = tile_col_mask(cola);
    return (a &~ ma) | (((b >> (8*colb)) << (8*cola)) & ma);
}

/** Set a[rowa] to the given data, and return the resulting tile. */
static inline tile_t tile_set_row(tile_t a, int rowa, tile_edge_t data) {
    /* spread data out into a row
     * the constant 0x2040810204081 has bits separated by 7, so it maps
     * 1->8, 2->16, 3->24, 4->32, 5->40, 6->48, 7->56
     * Bit 0 is dealt with by the |; otherwise it would collide.
     */
    tile_t dat = (tile_t)(data & 0xFE) * (tile_t)0x2040810204081ull | data;
    dat &= 0x0101010101010101ull;
    dat <<= rowa;

    return (a &~tile_row_mask(rowa)) | dat;
}

/** Swap rowa and rowb within a tile, and return the result.  If rowa==rowb, just return the input. */
static inline tile_t tile_swap_rows(tile_t a, int rowa, int rowb) {
    tile_t ret = a &~ tile_row_mask(rowa) &~ tile_row_mask(rowb);
    ret |= tile_row_mask(rowb) & (a >> rowa) << rowb;
    ret |= tile_row_mask(rowa) & (a >> rowb) << rowa;
    return ret;
}

/** Given pointers to two tiles, swap nrows starting with pa[rowa] with pb[rowb] */
static inline void tile2_bulk_swap_rows(tile_t *pa, tile_t *pb, int rowa, int rowb, int nrows) {
    tile_t a = *pa, b = *pb;
    tile_t ma = tile_row_bulk_mask(rowa, nrows), mb = tile_row_bulk_mask(rowb, nrows);
    *pa = (a&~ma) | ((b&mb)>>rowb) << rowa;
    *pb = (b&~mb) | ((a&ma)>>rowa) << rowb;
}

/** Copy nrows starting from b[rowb] to a[rowa] and return the new value of a. */
static inline tile_t tile_bulk_copy_rows(tile_t a, tile_t b, int rowa, int rowb, int nrows) {
    // copy rows from b[rowb] to a[rowa] and return a
    tile_t mask = tile_row_bulk_mask(rowa, nrows);
    return (a &~ mask) | (((b >> rowb) << rowa) & mask);
}

/** Xor nrows starting from b[rowb] to a[rowa] and return the new value of a. */
static inline tile_t tile_bulk_xor_rows(tile_t a, tile_t b, int rowa, int rowb, int nrows) {
    tile_t mask = tile_row_bulk_mask(0, nrows);
    return a ^ (((b >> rowb) & mask) << rowa);
}

/** Copy columns starting from b[colb] to a[cola] and return the new value of a. */
static inline tile_t tile_bulk_copy_cols(tile_t a, tile_t b, int cola, int colb, int ncols) {
    // swap cola and colb of the tile, and return it
    tile_t mask = tile_col_bulk_mask(cola, ncols);
    return (a &~ mask) | (((b >> (8*colb)) << (8*cola)) & mask);
}

/** Swap rows starting from a[rowa] to a[rowb] and return the new value of a.
 * The rows must not overlap.
 */
static inline tile_t tile_bulk_swap_rows(tile_t a, int rowa, int rowb, int nrows) {
    tile_t ma = tile_row_bulk_mask(rowa, nrows), mb = tile_row_bulk_mask(rowb, nrows);
    tile_t ret = a &~ ma &~ mb;
    ma &= a; mb &= a;
    return ret | ((ma>>rowa)<<rowb) | ((mb>>rowb)<<rowa);
}

/* Mask of all cols < col.  Used to clip data from random matrices */
static inline tile_t tile_mask_of_cols_less_than(int col) {
    return tile_col_bulk_mask(0,col);
}

/* Mask of all rows < rows.  Used to clip data from random matrices */
static inline tile_t tile_mask_of_rows_less_than(int row) {
    return tile_row_bulk_mask(0,row);
}

/* Return 1 if the tile is zero. */
static inline int tile_is_zero(tile_t a) { return a==0; }

/** Get the index of the least-significant nonzero entry of the tile.
 * Use bit_index_to_row and bit_index_to_col to parse it.
 * Returns 64 if no bits are set.
 */
static inline int tile_get_first_nonzero_entry(tile_t tile) { return ctz(tile); }

/** Obtain the row of an index */
static inline int bit_index_to_row(int index) { return index % 8; }

/** Obtain the col of an index */
static inline int bit_index_to_col(int index) { return index / 8; }

/*****************************************************
 * Operations on vectors of tiles
 *****************************************************/
#if __AVX2__ && !defined(TILE_NO_VECTOR)
    #include <immintrin.h>
    #define TILE_VECTOR_LENGTH 4
    typedef __m256i tile_vector_t;

    typedef struct {
        __m256i table_low, table_high;
    } tile_precomputed_t;

    /** Compute a table for accelerated multiplication by a */
    static inline tile_precomputed_t tile_precompute_vmul(tile_t a) {
        /* Miniature "four Russians" strategy.
        * Make a table for how upper/lower nibbles transform, and then use vpshufb
        * to shuffle.
        */
        __m128i av = _mm_set1_epi64x(a);
        __m256i one = _mm256_set1_epi16(0x101);

        // create permutation table
        const __m256i index = _mm256_set_epi64x(
            0xF0E0D0C0B0A09080ull,
            0x7060504030201000ull,
            0x0F0E0D0C0B0A0908ull,
            0x0706050403020100ull
        );

        __m256i table = _mm256_setzero_si256();
        for (unsigned i=0; i<8; i++) {
            __m256i t = _mm256_slli_epi16(one,i);
            table ^= _mm256_broadcastb_epi8(av) & _mm256_cmpeq_epi8(index & t, t);
            av = _mm_bsrli_si128(av,1);
        }
        tile_precomputed_t ret = {
            _mm256_permute2x128_si256(table,table,0x00),
            _mm256_permute2x128_si256(table,table,0x11)
        };
        return ret;
    }

    /** Multiply a tile (represented by a precomputed table) by a vector of tiles */
    static inline tile_vector_t tile_vmul (const tile_precomputed_t *pa, tile_vector_t b) {
        __m256i low = _mm256_set1_epi16(0xF0F);
        return _mm256_shuffle_epi8(pa->table_low,  b&low)
             ^ _mm256_shuffle_epi8(pa->table_high, _mm256_srli_epi16(b,4)&low);
    }

    /** Read a possibly-unaligned tile */
    static inline tile_vector_t tile_read_v(const tile_t *pa) {
        return _mm256_loadu_si256((const tile_vector_t*)pa);
    }

    /** Write a possibly-unaligned tile */
    static inline void tile_write_v(tile_t *pa, tile_vector_t a) {
        _mm256_storeu_si256((tile_vector_t*)pa, a);
    }

    /** Read 1 <= n <= TILE_VECTOR_LENGTH possibly unaligned tiles. */
    static inline tile_vector_t tile_read_vpartial(const tile_t *pa, int n) {
        __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n), _mm256_set_epi64x(3,2,1,0));
        return _mm256_maskload_epi64((const long long*)pa, mask);
    }

    /** Write 1 <= n <= TILE_VECTOR_LENGTH possibly unaligned tiles. */
    static inline void tile_write_vpartial(tile_t *pa, int n, tile_vector_t a) {
        __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x(n), _mm256_set_epi64x(3,2,1,0));
        _mm256_maskstore_epi64((long long*)pa, mask, a);
    }
#elif (__ARM_NEON__ || __ARM_NEON) && !defined(TILE_NO_VECTOR)
    #include <arm_neon.h>
    #define TILE_VECTOR_LENGTH 2
    typedef uint8x16_t tile_vector_t;

    typedef struct {
        uint8x16_t table_low, table_high;
    } tile_precomputed_t;

    /** Compute a table for accelerated multiplication by a */
    static inline tile_precomputed_t tile_precompute_vmul(tile_t a) {
        uint8x8_t av = vreinterpret_u8_u64(vdup_n_u64(a));
        uint8x16_t one = vdupq_n_u8(1);

        // create permutation table
        const uint8_t index[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        const uint8x16_t index_low = vld1q_u8(index);
        const uint8x16_t index_high = vshlq_n_u8(index_low,4);

        uint8x16_t low = vdupq_n_u8(0), high = vdupq_n_u8(0);
        for (unsigned i=0; i<8; i++) {
            low  ^= vdupq_lane_u8(av,0) & vceqq_u8(index_low  & one, one);
            high ^= vdupq_lane_u8(av,0) & vceqq_u8(index_high & one, one);
            one = vshlq_n_u8(one,1);
            av = vreinterpret_u8_u64(vshr_n_u64(vreinterpret_u64_u8(av),8));
        }
        tile_precomputed_t ret = { low, high };
        return ret;
    }

    /** Multiply a tile (represented by a precomputed table) by a vector of tiles */
    static inline tile_vector_t tile_vmul (const tile_precomputed_t *pa, tile_vector_t b) {
        uint8x16_t low = vdupq_n_u8(0xF);
        return vqtbl1q_u8(pa->table_low,  b&low)
             ^ vqtbl1q_u8(pa->table_high, vshrq_n_u8(b,4));
    }

    /** Read a possibly-unaligned tile */
    static inline tile_vector_t tile_read_v(const tile_t *pa) {
        return vld1q_u8((const uint8_t*)pa);
    }

    /** Write a possibly-unaligned tile */
    static inline void tile_write_v(tile_t *pa, tile_vector_t a) {
        vst1q_u8((uint8_t*)pa, a);
    }

    /** Read 1 <= n <= TILE_VECTOR_LENGTH a possibly-unaligned tiles */
    static inline tile_vector_t tile_read_vpartial(const tile_t *pa, int n) {
        if (n >= 2) return tile_read_v(pa);
        uint64x2_t ret = {0,0};
        return vreinterpretq_u64_u8(vld1q_lane_u64(pa,ret,0));
    }

    /** Write 1 <= n <=  TILE_VECTOR_LENGTH a possibly-unaligned tiles */
    static inline void tile_write_vpartial(tile_t *pa, int n, tile_vector_t a) {
        if (n >= 2) tile_write_v(pa,a);
        else vst1q_lane_u64(pa,vreinterpretq_u8_u64(a),0);
    }
#else // !__AVX2__ && !__ARM_NEON__
    /* Scalar implementation */
    #define TILE_VECTOR_LENGTH 1
    typedef tile_t tile_vector_t, tile_precomputed_t;
    static inline tile_precomputed_t tile_precompute_vmul(tile_t a) { return a; }
    static inline tile_t tile_vmul(const tile_precomputed_t *pa, tile_vector_t b) {
        return tile_mul(*pa,b);
    }

    static inline tile_vector_t tile_read_v(const tile_t *pa) { return *pa; }
    static inline void tile_write_v(tile_t *pa, tile_vector_t a) { *pa=a; }
    static inline tile_vector_t tile_read_vpartial(const tile_t *pa, int n) { (void)n; return *pa; }
    static inline void tile_write_vpartial(tile_t *pa, int n, tile_vector_t a) { (void)n; *pa=a; }
#endif // __AVX2__

#endif // __TILE_OPS_H__
