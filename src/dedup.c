/**
 * @file dedup.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Giant hack of a deduplicator.  You'd think a data structures library
 * would have better.
 */

#include <errno.h>
#include "dedup.h"

/** Compare two lfr_nonuniform_relations, with the given ordering of responses */
static int cmp_relation(const void *avoid, const void *bvoid, int yes_overrides_no) {
    const lfr_nonuniform_relation_t
        *a = *(const lfr_nonuniform_relation_t *const*)avoid,
        *b = *(const lfr_nonuniform_relation_t *const*)bvoid;
    
    /* Sort by query length */
    if (a->query_length > b->query_length) return 1;
    if (a->query_length < b->query_length) return -1;
    
    /* Sort by query data */
    int ret = memcmp(a->query,b->query,a->query_length);
    if (ret) return ret;

    /* Sort by response */
    if (b->response > a->response) return  yes_overrides_no;
    if (b->response < a->response) return -yes_overrides_no;
    
    return 0;
}

/* Compare: yes overrides no */
static int cmp_relation_yon(const void *avoid, const void *bvoid) {
    return cmp_relation(avoid,bvoid,1);
}

/* Compare: no overrides yes */
static int cmp_relation_noy(const void *avoid, const void *bvoid) {
    return cmp_relation(avoid,bvoid,-1);
}

/* N-hash functions for Bloom filters */
static inline __attribute__((always_inline)) void bloom_hash (
    uint64_t *out,
    int n_out,
    const uint8_t *data,
    size_t data_length,
    uint64_t seed
) {
    hash_result_t hash_res = murmur3_x64_128_extended_seed(data, data_length, seed);
    for (int j=0; j<n_out; j++) {
        out[j] = hash_res.low64;
        // just make up an expander
        hash_res.low64 ^= rotl64(hash_res.high64, 37);
        hash_res.high64 += rotl64(hash_res.low64, 8);
    }
}

/** Use a counting Bloom filter as a first pass to uniquify a pair of lists.
 * Returns a bitset of relations which may be duplicates, according to the Bloom
 * filter.
 *
 * @todo: this section needs unit tests.
 */
static bitset_t bloom_unique (
    const bitset_t relevant,
    const lfr_nonuniform_relation_t *rel,
    size_t nrel
) {
    #define NHASHES 4
    uint64_t hashes[NHASHES];
    size_t elements = bitset_popcount(relevant, nrel) * NHASHES * 4;

    /* Make it a power of 2 */
    while (elements & (elements-1)) {elements &= elements-1; }
    size_t mask = 2*(elements-1); // low bit is clear
    
    /* 2-bit counting bloom filter */
    size_t btsize = ((elements * 2 + 63)/64) * 8 ;
    uint64_t *bloom_table = malloc(btsize);
    if (bloom_table == NULL) {
        return NULL;
    }
    
    bitset_t potential_dups = bitset_duplicate(relevant, nrel);
    if (potential_dups == NULL) {
        free(bloom_table);
        return NULL;
    }

    /* The more values scanned be the Bloom filter, the more
     * false positives there will be.  Conversely, once we have reduced
     * the number of possible duplicates, the lower the false positive rate.
     * So we perform several tours through the array, each time with fewer
     * candidates and a (hopefully) lower FPR.  The algorithm bails after
     * 5 tours (totally arbitrary) or when the number of candidates stops
     * decreasing.
     */
    const int NTOURS = 5;
    size_t hits_prev = 0;
    uint64_t arbitrary_seed = 0x8a712e241b1c2791; /* Arbitrary.  TODO: randomize? */
    for (int tour = 0; tour < NTOURS; tour++) {
        memset(bloom_table, 0, btsize);
        
        /* Insert them into the bloom table */
        for (size_t i=0; i<nrel; i++) {
            if (!bitset_test_bit(potential_dups, i)) {
                continue;
            }
            bloom_hash(hashes, NHASHES, rel[i].query, rel[i].query_length, arbitrary_seed);
            for (int j=0; j<NHASHES; j++) {
                size_t h = hashes[j] & mask;
                uint64_t b = bloom_table[h/64];
                
                /* Add 1 and saturate at 2 */
                b += (uint64_t)1 << (h%64);
                b &= ~(b>>1 & 0x5555555555555555ull);
                bloom_table[h/64] = b;
            }
        }
    
        /* Count the number of collisions */
        size_t hits = 0;
        for (ssize_t i=nrel-1; i>=0; i--) {
            if (!bitset_test_bit(potential_dups, i)) continue;
            bloom_hash(hashes, NHASHES, rel[i].query, rel[i].query_length, arbitrary_seed);
            uint64_t in = 1;
            for (unsigned j=0; j<NHASHES && in; j++) {
                size_t h = hashes[j] & mask;
                in = bloom_table[h/64] & (2ull << (h%64));
            }
        
            if (in) {
                hits++;
            } else {
                bitset_clear_bit(potential_dups, i);
            }
        }
        if (hits == 0 || hits == hits_prev) break;
        hits_prev = hits;
    }
    
    free(bloom_table);
    return potential_dups;
}

/** Allocate and return a bitset indicating which relations are duplicates
 * and should be ignored.  One of each set of duplicate values will be kept
 * (not marked as a duplicate) and the rest will be discarded (marked).
 *
 * @param yes_overrides_no Controls how relations with the same key are handled.  If < 0, the least value will be taken.
 * If > 0, the greatest value will be taken.  If 0, relations with different values are an error.
 * @param[out] n_out_p A pointer to place the number of duplicate entries.
 * @param[out] out The bitset of duplicates will be allocated here.
 *
 * @return 0 on success.
 * @return -ENOMEM on out of memory.
 * @return -EEXIST if the same key has two different values and yes_overrides_no == 0.
 *
 * @todo: this section needs unit tests.
 */

int remove_duplicates (
    bitset_t relevant,
    size_t *n_out_p,
    const lfr_nonuniform_relation_t *rel,
    size_t nrel,
    int yes_overrides_no
) {
    /* Strategy: first use a counting Bloom filter to make candidate dupes.
     * Create pointers to all candidates and then sort the pointers (because
     * the actual objects are const), in an order that depends on whether
     * yes_overrides_no or vice versa.  Then pass through and mark all but
     * the first as a duplicate.
     */
    *n_out_p = 0;
    bitset_t potential_dups = bloom_unique(relevant, rel, nrel);
    if (potential_dups == NULL) {
        return -ENOMEM;
    }
    
    size_t ndups = bitset_popcount(potential_dups, nrel);
    const lfr_nonuniform_relation_t **idxs = malloc(sizeof(*idxs)*ndups);
    if (idxs == NULL) {
        free(potential_dups);
        return -ENOMEM;
    }
    
    ssize_t start = 0;
    for (size_t i=0; i<ndups; i++) {
        start = bitset_next_bit(potential_dups,nrel,start);
        assert(start >= 0);
        idxs[i] = &rel[start];
        start++;
    };

    qsort(idxs, ndups, sizeof(*idxs),
        (yes_overrides_no>0) ? cmp_relation_yon : cmp_relation_noy);
    bitset_destroy(potential_dups);
    
    size_t n_out = 0;
    
    const lfr_nonuniform_relation_t *prev = NULL;
    
    for (size_t i=0; i<ndups; i++) {
        if (prev && !memcmp(prev->query, idxs[i]->query, prev->query_length)) {
            bitset_clear_bit(relevant, idxs[i] - rel);
            n_out++;
            int response_differs = prev->response != idxs[i]->response;
            if (yes_overrides_no==0 && response_differs != 0) {
                free(idxs);
                return -EEXIST;
            }
        } else {
            prev = idxs[i];
        }
    }
    
    free(idxs);
    *n_out_p = n_out;

    return 0;
}
