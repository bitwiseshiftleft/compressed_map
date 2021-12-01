/** @file frayed_ribbon.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Matrix shape plugin for frayed ribbon matrices.
 * Modify or make another .c/.o to test other matrix shapes.
 */
#include "lfr_uniform.h"
#include "util.h"

#ifndef LFR_OVERPROVISION
#define LFR_OVERPROVISION 1024
#endif

static const size_t EXTRA_ROWS = 8;
size_t API_VIS _lfr_uniform_provision_columns(size_t rows) {
    size_t cols = rows + EXTRA_ROWS;
#if LFR_OVERPROVISION
    cols += cols/LFR_OVERPROVISION;
#endif
    cols += (-cols) % (8*LFR_BLOCKSIZE);
    if (cols <= 8*LFR_BLOCKSIZE) cols = 16*LFR_BLOCKSIZE;
    return cols;
}

size_t API_VIS _lfr_uniform_provision_max_rows(size_t cols) {
    size_t cols0 = cols;
    (void)cols0;
#if LFR_OVERPROVISION
    cols -= cols / (LFR_OVERPROVISION+1);
#endif
    if (cols <= EXTRA_ROWS) return 0;
    if (_lfr_uniform_provision_columns(cols-EXTRA_ROWS) > cols0) cols--;
    assert(_lfr_uniform_provision_columns(cols-EXTRA_ROWS) <= cols0);
    return cols - EXTRA_ROWS;
}

typedef struct {
    uint8_t stride_seed[4];
    uint8_t a_seed[4];
} lfr_uniform_sampler_t;
const unsigned int LFR_META_SAMPLE_BYTES = sizeof(lfr_uniform_sampler_t);

/** The main function: given a seed, sample block positions */
_lfr_uniform_row_indices_s _lfr_uniform_sample_block_positions (
    size_t nblocks,
    const uint8_t *seed_bytes
) {
    /* Parse the seed into uints */
    const lfr_uniform_sampler_t *seed = (const lfr_uniform_sampler_t*) seed_bytes;
    uint64_t stride_seed = le2ui(seed->stride_seed, sizeof(seed->stride_seed));
    uint64_t a_seed      = le2ui(seed->a_seed,      sizeof(seed->a_seed));
    __uint128_t nblocks_huge = nblocks;

    // calculate log(nblocks)<<48 in a smooth way
    uint64_t k = high_bit(nblocks);
    uint64_t smoothlog = (k<<48) + (nblocks<<(48-k)) - (1ull<<48);
    uint64_t leading_coefficient = (12ull<<8)/LFR_BLOCKSIZE; // experimentally determined
    uint64_t num = smoothlog * leading_coefficient; 

#if LFR_OVERPROVISION
    stride_seed |= (1ull<<33) / LFR_OVERPROVISION; // | instead of + because it can't overflow
#endif
    uint64_t den = ((stride_seed * stride_seed) * nblocks_huge) >> 32;
#if (!LFR_OVERPROVISION) || (LFR_OVERPROVISION > 1ull<<16)
    den++; // to prevent it from being 0
#endif
    uint64_t b_seed = (num / den + a_seed) & 0xFFFFFFFF; // den can't be 0 because stride_seed is adjusted

    uint64_t a = (a_seed * nblocks_huge)>>32, b = (b_seed * nblocks_huge)>>32;
    if (a==b && ++b >= nblocks) b=0;
    
    _lfr_uniform_row_indices_s out = {a,b};
    return out;
}
