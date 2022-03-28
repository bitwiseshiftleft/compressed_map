/** @file lfr_nonuniform.c
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Nonuniform maps implementation.
 */
#include <math.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <sys/random.h>
#include "lfr_nonuniform.h"
#include "lfr_uniform.h"
#include "bitset.h"
#include "util.h"

#define LFR_INTERVAL_BYTES (40/8)
#define LFR_INTERVAL_SH (8*(sizeof(lfr_locator_t) - LFR_INTERVAL_BYTES))
#define LFR_INTETRVAL_TOTAL ((lfr_locator_t)1 << (8*LFR_INTERVAL_BYTES))

/** Determine where phases need to start for the given interval bounds */
static lfr_locator_t lfr_nonuniform_summarize_plan (
    const lfr_nonuniform_intervals_t *response_map,
    unsigned nitems
) {
    lfr_locator_t plan = 0;
    for (unsigned i=0; i<nitems; i++) {
        lfr_locator_t width = response_map[(i+1)%nitems]->lower_bound - response_map[i]->lower_bound;
        if (!width) continue;
        int hi = high_bit(width);
        plan |= (lfr_locator_t)1 << hi;
        if (width & (width-1)) {
            plan |= (lfr_locator_t)2 << hi;
        }
    }
    return plan;
}

/** Structure for interval optimizer */
typedef struct {
    size_t count;                   /** How many times the value appears in the dict */
    lfr_locator_t width;            /** The width of the locator interval: value to be optimized */
    lfr_response_t resp;            /** The response / dictionary value itself */
} formulation_item_t;

/** Sort by how much was rounded off of the interval size,
 * proportionally, descending.
 */
static int cmp_fit(const void *va, const void *vb) {
    const formulation_item_t *a = va, *b = vb;
    __uint128_t ascore = a->width, bscore = b->width;
    ascore *= b->count; bscore *= a->count;
    if (ascore < bscore) return -1;
    if (bscore < ascore) return  1;
    /* Arbitrary order for the rest.  Smaller intervals
     * first to decrease phases. */
    if (a->width < b->width) return -1;
    if (a->width > b->width) return 1;
    if (a->count < b->count) return -1;
    if (a->count > b->count) return 1;
    if (a->resp < b->resp) return -1;
    if (a->resp > b->resp) return 1;
    return 0;
}

/** Sort by alignment */
static int cmp_align(const void *va, const void *vb) {
    const formulation_item_t *a = va, *b = vb;
    lfr_locator_t wa = a->width, wb = b->width;
    wa &= ~(wa-1); wb &= ~(wb-1);
    if (wa > wb) return -1;
    if (wa < wb) return  1;
    if (a->width < b->width) return -1;
    if (a->width > b->width) return 1;
    if (a->count < b->count) return -1;
    if (a->count > b->count) return 1;
    if (a->resp < b->resp) return -1;
    if (a->resp > b->resp) return 1;
    return 0;
}

static inline lfr_locator_t floor_power_of_2(lfr_locator_t x) {
    if (x==0) return 0;
    return (lfr_locator_t) 1 << high_bit(x);
}

/**
 * Given the counts for each value, optimize the intervals assigned to each
 * and compute a "plan" measuring where the phases begin and end.
 */
static lfr_locator_t lfr_nonuniform_formulate_plan (
    lfr_nonuniform_intervals_t *response_map,
    formulation_item_t *items,
    unsigned nitems
) {
    /* Special cases: no items */
    if (nitems == 0) {
        return 0; /* Don't need any phases of course */
    } else if (nitems == 1) {
        response_map[0]->lower_bound = 0;
        response_map[0]->response = items[0].resp;
        return 0; /* Still don't need any phases */
    }

    /* Count total items */
    size_t total = 0;
    for (unsigned i=0; i<nitems; i++) {
        total += items[i].count;
    }
    assert(total > 0);
    
    /* How to optimize interval widths:
     * 1. Assign to each locator i the fraction pi/total, rounded
     *    down to a power of 2.
     *    (i.e. interval width floor_po2(pi*LFR_INTERVAL_TOTAL / total)).
     * 2. Count how much total interval width is left over.
     * 3. Sort the intervals by badness of fit, i.e. by pi/interval_i descending.
     * 4. Starting with the worst fit, increase the width to the next power of 2,
     *    or the left-over width, until the left-over width is used up.
     */

    lfr_locator_t total_width = 0;
    for (unsigned i=0; i<nitems; i++) {
        __uint128_t count = items[i].count;
        lfr_locator_t width = floor_power_of_2((count << (8*sizeof(lfr_locator_t))) / total);
        items[i].width = width;
        if (LFR_INTERVAL_BYTES < sizeof(lfr_locator_t)) {
            // Can't support maps where the ratio is trillions : 1.
            assert(width >> LFR_INTERVAL_SH > 0);
        }
        total_width += width;
    }
    qsort(items,nitems,sizeof(items[0]),cmp_fit);

    /* remaining width = "entire interval" (=0) - total_width; */
    lfr_locator_t remaining_width = -total_width;

    /* While there is remaining width, widen the intervals in priority order */
    for (unsigned i=0; remaining_width > 0; i++) {
        if (i >= nitems) {
            assert(0 && "bug: didn't exhaust the width");
            return 0;
        }
        if (items[i].width >= remaining_width) {
            items[i].width += remaining_width;
            remaining_width = 0; /* and thus, break */
        } else {
            remaining_width -= items[i].width;
            items[i].width *= 2;
        }
    }

    /* Sort and calculate the interval bounds. */
    qsort(items,nitems,sizeof(items[0]),cmp_align);
    total = 0;
    for (unsigned i=0; i<nitems; i++) {
        response_map[i]->lower_bound = total;
        response_map[i]->response = items[i].resp;
        total += items[i].width;
    }
    
    /* Calculate the plan, which is a bitmask of where the phases begin/end */
    return lfr_nonuniform_summarize_plan(response_map, nitems);
}

/* Give a target number of constraints for each phase, to reroll if the map would
 * be much larger than expected.
 */
static void get_nconstr_targets (
    size_t *targets,
    lfr_locator_t plan,
    unsigned nphases,
    unsigned nitems,
    const formulation_item_t *items
) {
    /* False positives must be at most FP_TIGHTNESS_MULT * expected + FP_TIGHTNESS_ADD */
    const float FP_TIGHTNESS_MULT = 1.005; 
    const int FP_TIGHTNESS_ADD = 32;

    for (unsigned phase=0; plan; plan &= plan-1, phase++) {
        if (phase >= nphases) {
            assert(0 && "counted the phases wrong in get_nconstr_targets");
            break;
        }
        lfr_locator_t plan_interval = plan &~ (plan-1);
        size_t total=0;
        for (unsigned i=0; i<nitems; i++) {
            lfr_locator_t interval_width = items[i].width;
            size_t count = items[i].count;
            if (interval_width <= plan_interval) {
                total += count;
            } else if (interval_width/2 <= plan_interval) {
                double frac = 2 - (double)interval_width / plan_interval;
                total += frac*count * FP_TIGHTNESS_MULT + FP_TIGHTNESS_ADD;
            }
        }
        targets[phase] = total;
    }
}

/** Binary search for item in map */
static lfr_response_t bsearch_bound (
    unsigned nitems,
    const lfr_nonuniform_intervals_t *items,
    lfr_locator_t loc
) {
    unsigned lower = 0, upper = nitems-1;
    while (lower < upper) {
        unsigned mid = (lower+upper+1)/2;
        lfr_locator_t midval = items[mid]->lower_bound;
        if (loc >= midval) {
            lower = mid;
        } else {
            upper = mid-1;
        }
    }
    return items[lower]->response;
}

/* Return nonzero iff constrained */
static inline int constrained_this_phase (
    lfr_locator_t *constraint,
    int phlo,
    int phhi,
    lfr_locator_t cur,
    lfr_locator_t lowx, // exclusive
    lfr_locator_t high  // inclusive
) {        
    lfr_locator_t
        lo_phase_bit = (lfr_locator_t)1<<phlo,
        hi_phase_bit = (lfr_locator_t)1<<phhi,
        mask_before = lo_phase_bit-1,
        mask_after = 2*hi_phase_bit-1;
    
    cur &= mask_before;
    if (high - lowx == 0 || high-lowx > mask_after) {
        /* The interval is wider than the phase, so it doesn't matter */
        return 0;
    }
    
    high = (high-cur) & mask_after;
    lowx = (lowx-cur) & mask_after;
    if (high > lowx || high>>phlo != lowx>>phlo) {
        *constraint = high>>phlo;
        return 1;
    } else {
        /* The interval wraps, and includes all multiples of 1<<phlo */
        return 0;
    }
}

#ifndef LFR_PHASE_TRIES
#define LFR_PHASE_TRIES 5
#endif

int lfr_nonuniform_count_items (
    size_t *nitems_p,
    formulation_item_t **items_p,
    const lfr_builder_t nonu_builder
) {
    const size_t INITIAL_NITEMS = 2048;
    size_t nrelns = nonu_builder->used, nitems=0;
    *items_p = NULL;
    *nitems_p = 0;

    formulation_item_t *items = NULL;

    lfr_builder_t hashtable_for_counting;
    int ret = lfr_builder_init(hashtable_for_counting, INITIAL_NITEMS,
        INITIAL_NITEMS*sizeof(lfr_response_t), 0);
    if (ret) return ret;

    /* Insert into the hashtable */
    for (size_t i=0; i<nrelns; i++) {
        lfr_response_t *count = lfr_builder_lookup_insert(
            hashtable_for_counting,
            (const uint8_t*) &nonu_builder->relations[i].value,
            sizeof(nonu_builder->relations[i].value),
            0
        );
        if (count == NULL) {
            ret = ENOMEM;
            goto done;
        }
        (*count)++;
    }

    /* Allocate the items */
    nitems = hashtable_for_counting->data_used / sizeof(lfr_response_t);
    if (nitems == 0) {
        /* Can't create a map with no items: it's a footgun, even more than
         * other uses of this library.
         */
        return EINVAL;
    }
    *items_p = items = calloc(nitems,sizeof(*items));
    if (items == NULL && nitems > 0) {
        ret = ENOMEM;
        goto done;
    }
    *nitems_p = nitems;

    /* Pull them out of the hashtable */
    const uint8_t *data_i = hashtable_for_counting->data;
    for (size_t i=0; i<nitems; i++, data_i += sizeof(lfr_response_t)) {
        memcpy(&items[i].resp, data_i, sizeof(lfr_response_t));

        lfr_response_t *count = lfr_builder_lookup(hashtable_for_counting,
            data_i, sizeof(lfr_response_t));
        assert(count != NULL);
        assert(*count > 0);
        items[i].count = *count;
    }

    ret = 0;

done:
    lfr_builder_destroy(hashtable_for_counting);

    return ret;
}

/* Create a lfr_nonuniform. */
int API_VIS lfr_nonuniform_build (
    lfr_nonuniform_map_t out,
    const lfr_builder_t nonu_builder
) { 
    /*************************************************************
     * Setup and counting phase.
     * Count the items, formulate a plan, allocate space, create headers.
     *************************************************************/

    /* Preinitialize so that we can goto done */
    memset(out,0,sizeof(*out));
    int ret = -1;
    size_t nrelns = nonu_builder->used, nitems;
    formulation_item_t *items = NULL;
    size_t *target_constraints = NULL;
    lfr_locator_t plan;
    const lfr_relation_t *relns = nonu_builder->relations;
    lfr_locator_t *current = NULL;
    bitset_t relevant = NULL;
    int *response_perm = NULL;

    int *phase_salt = NULL;
    
    ret = lfr_nonuniform_count_items(&nitems, &items, nonu_builder);
    if (ret) goto done;
    
    /* Create the response map */
    out->response_map = calloc(nitems, sizeof(*out->response_map));
    if (out->response_map == NULL) goto alloc_failed;
    out->nresponses = nitems;

    /* Create plan and interval bounds */
    out->plan = plan = lfr_nonuniform_formulate_plan(out->response_map, items, nitems);
    int nphases = popcount(plan);
    target_constraints = calloc(nphases, sizeof(*target_constraints));
    if (target_constraints == NULL) goto alloc_failed;
    get_nconstr_targets(target_constraints, plan, nphases, nitems, items);

    /* Create the permutation of responses */
    response_perm = calloc(nitems, sizeof(*response_perm));
    if (response_perm == NULL) goto alloc_failed;
    for (unsigned i=0; i<nitems; i++) {
        response_perm[out->response_map[i]->response] = i;
    }

    // Allocate the phase data
    out->phases = calloc(nphases, sizeof(*out->phases));
    if (out->phases == NULL) goto alloc_failed;
    out->nphases = nphases;
    phase_salt = calloc(nphases+1, sizeof(*phase_salt)); // +1 so we can be lazy
    if (phase_salt == NULL) goto alloc_failed;

    // Allocate set of currently-relevent relations
    relevant = bitset_init(nrelns);
    if (relevant == NULL) goto alloc_failed;

    // Allocate current value of each item's locator
    current = calloc(nrelns, sizeof(*current));
    if (current == NULL) goto alloc_failed;
    
    lfr_builder_t builder;
    memset(builder,0,sizeof(builder));

    // Search tree for suitable salts
    phase_salt[0] = 0;
    int phase=0;
    for (int try=0; phase >= 0 && phase < nphases && try < nphases + builder->max_tries; try++) {
        phase_salt[phase]++;

        // Search heuristic: If we've retried this phase several times without success,
        // abort it and back up to previous phase
        while (phase > 0 && phase_salt[phase] > LFR_PHASE_TRIES) {
            phase--;
            phase_salt[phase]++;
        }

        /* Get the number of value-bits to be determined this phase */
        lfr_locator_t plan_tmp = plan;
        for (int i=0; i<phase; i++) {
            plan_tmp &= plan_tmp-1;
        }
        int phlo = ctz(plan_tmp);
        int phhi = ctz(plan_tmp & (plan_tmp-1)) - 1;
        
        /* Find the target number of rows. */
        size_t nconstraints = 0;
        
        /* Count the number of constrained rows */
        bitset_clear_all(relevant, nrelns);
        for (size_t i=0; i<nrelns; i++) {
            int resp = response_perm[relns[i].value];
            lfr_locator_t
                lowx = out->response_map[resp]->lower_bound-1,
                high = out->response_map[(resp+1) % nitems]->lower_bound-1,
                cur = current[i],
                ignored;
                
            if (constrained_this_phase(&ignored, phlo, phhi, cur, lowx, high)) {
                nconstraints ++;
                bitset_set_bit(relevant,i);
            }
        }
        
        if (nconstraints > target_constraints[phase]) {
            // Too many false positives from previous phase
            phase--;
            continue;
        }
        
        /* Create the builder */
        lfr_builder_destroy(builder);

        ret = lfr_builder_init(builder, nconstraints, 0, LFR_NO_COPY_DATA | LFR_NO_HASHTABLE); // no salt yet, set in iteration
        if (ret) { goto done; }
        if (phase==0) {
            if (( ret = getentropy(&builder->salt, sizeof(builder->salt)) )) goto done;
        } else {
            builder->salt = out->phases[phase-1]->salt;
        }
        builder->salt_hint = phase_salt[phase];
        builder->max_tries = LFR_PHASE_TRIES;

        /* Build the uniform map using constrained items */
        for (size_t i=0; i<nrelns; i++) {
            if (!bitset_test_bit(relevant, i)) continue; // it's not constrained this phase
            lfr_response_t resp = response_perm[relns[i].value];
        
            lfr_locator_t
                lowx = out->response_map[resp]->lower_bound-1,
                high = out->response_map[(resp+1) % nitems]->lower_bound-1,
                cur = current[i],
                constraint;
                                
            int c = constrained_this_phase(&constraint, phlo, phhi, cur, lowx, high);
            assert(c);
            lfr_builder_insert(builder, relns[i].key, relns[i].keybytes, constraint);
        }
        
        lfr_uniform_map_destroy(out->phases[phase]);
        int phase_ret = lfr_uniform_build(out->phases[phase], builder, phhi+1-phlo);

        if (phase_ret == 0 && phase < nphases-1) {
            /* It's not the last phase.  Adjust the values of the items.
             * Skip the ones that are powers of 2 in size: they are automatically
             * considered in that phase and later.
             *
             * Unfortunately, the non-power-of-2 class is usually the largest.
             * TODO: try to skip the query in cases where it doesn't matter?
             */
            for (size_t i=0; i<nrelns; i++) {
                unsigned r = response_perm[relns[i].value];
                lfr_locator_t w = ((r == nitems-1) ? 0 : out->response_map[r+1]->lower_bound)
                               - out->response_map[r]->lower_bound;
                if ((w & (w-1))==0) continue; // don't care about the result for the power-of-2 ones

                lfr_locator_t ci = current[i], mask=((lfr_locator_t)1<<phlo)-1;
                ci &= mask;
                ci += lfr_uniform_query(out->phases[phase], relns[i].key, relns[i].keybytes) << phlo;
                current[i] = ci;
            }
        }

        if (phase_ret == 0) {
            // Success!
            phase++;
            phase_salt[phase] = 0;
        }
    }
    
    if (phase < nphases) ret = EAGAIN;
    goto done;

alloc_failed:
    ret = ENOMEM;

done:
    /* Clean up all allocations */
    free(phase_salt);
    free(response_perm);
    bitset_destroy(relevant);
    free(current);
    lfr_builder_destroy(builder);
    free(items);
    free(target_constraints);
    if (ret != 0) lfr_nonuniform_map_destroy(out);
    
    return ret;
}
    
void API_VIS lfr_nonuniform_map_destroy(lfr_nonuniform_map_t map) {
    for (int i=0; i<map->nphases; i++) {
        lfr_uniform_map_destroy(map->phases[i]);
    }
    free(map->response_map);
    free(map->phases);
    memset(map,0,sizeof(*map));
}

#include <stdio.h>
lfr_response_t API_VIS lfr_nonuniform_query (
    const lfr_nonuniform_map_t map,
    const uint8_t *key,
    size_t keybytes
) {
    if (map->nphases <= 0) return map->response_map[0]->response;
    lfr_locator_t loc=0, plan=map->plan, known_mask = (plan-1) &~ plan;

    /* The upper bits are the most informative.  However, in most cases the second-highest
     * map has more bits than the highest one, so it's actually fastest to start there.
     */
    if (map->nphases >= 2) {
        int h1 = high_bit(plan);
        plan ^= (lfr_locator_t)1<<h1;
        int h2 = high_bit(plan);
        lfr_locator_t thisphase = lfr_uniform_query(map->phases[map->nphases-2], key, keybytes);

        known_mask |= ((lfr_locator_t)1<<h1) - ((lfr_locator_t)1<<h2);
        loc |= thisphase << h2;

        lfr_response_t lower = bsearch_bound(map->nresponses,map->response_map,loc);
        lfr_response_t upper = bsearch_bound(map->nresponses,map->response_map,loc |~ known_mask);
        if (upper == lower) return upper;
    }
    plan = map->plan;

    for (int phase=map->nphases-1; phase >= 0; phase--) {
        int h = high_bit(plan);
        plan ^= (lfr_locator_t)1<<h;
        if (phase == map->nphases - 2) continue;

        lfr_locator_t thisphase = lfr_uniform_query(map->phases[phase], key, keybytes);

        loc |= thisphase << h;
        known_mask |= -((lfr_locator_t)1<<h);
        
        lfr_response_t lower = bsearch_bound(map->nresponses,map->response_map,loc);
        lfr_response_t upper = bsearch_bound(map->nresponses,map->response_map,loc |~ known_mask);
        if (upper == lower) return upper;
    };
    
    assert(0 && "bug or map is corrupt: lfr_nonuniform_query should have narrowed down a response");
    return -(lfr_response_t)1;
}

#define LFR_NITEMS_BYTES  (32/8)
#define LFR_NBLOCKS_BYTES (32/8)

typedef struct {
    uint8_t plan[LFR_INTERVAL_BYTES];
    uint8_t nitems[LFR_NITEMS_BYTES];
    uint8_t file_salt[sizeof(lfr_salt_t)-1];
} __attribute__((packed)) lfr_nonuniform_header_t;

typedef struct {
    uint8_t lg_weight;
    uint8_t response[sizeof(lfr_response_t)];
} __attribute__((packed)) lfr_response_header_t;

typedef struct {
    uint8_t salt_hint;
    uint8_t nblocks[LFR_NBLOCKS_BYTES];
} __attribute__((packed)) lfr_phase_header_t;


size_t API_VIS lfr_nonuniform_map_serial_size(const lfr_nonuniform_map_t map) {
    size_t ret = sizeof(lfr_nonuniform_header_t);
    ret += map->nresponses * sizeof(lfr_response_header_t);
    ret += map->nphases * sizeof(lfr_phase_header_t);
    for (int i=0; i<map->nphases; i++) {
        ret += _lfr_uniform_map_vector_size(map->phases[i]);
    }
    return ret;
}

int API_VIS lfr_nonuniform_map_serialize(uint8_t *out, const lfr_nonuniform_map_t map) {
    lfr_nonuniform_header_t *header = (lfr_nonuniform_header_t*) out;

    /* Serialize map header */
    int ret = ui2le(header->nitems, sizeof(header->nitems), map->nresponses);
    if (ret) return ret;
    if (popcount(map->plan) != map->nphases) return EINVAL;
    ret = ui2le(header->plan, sizeof(header->plan), map->plan >> LFR_INTERVAL_SH);
    if (ret) return ret;
    ret = ui2le(header->file_salt, sizeof(header->file_salt), map->phases[0]->salt >> 8);
    if (ret) return ret;
    out += sizeof(*header);

    /* Serialize response data */
    lfr_locator_t base = 0;
    for (int i=0, seen_balance=0; i<map->nresponses; i++) {
        lfr_response_header_t *re = (lfr_response_header_t *)out;
        lfr_locator_t nxt = (i < map->nresponses-1) ? map->response_map[i+1]->lower_bound : 0;
        lfr_locator_t width = nxt-base;
        base = nxt;

        if (width == 0 && map->nresponses > 1) {
            return EINVAL;
        } else if (width > 0 && (width & (width-1)) == 0) {
            re->lg_weight = high_bit(width);
        } else {
            /* It's the odd one out */
            if (seen_balance) return EINVAL;
            re->lg_weight = 0xFF;
            seen_balance = 1;
        }
        ui2le(re->response, sizeof(re->response), map->response_map[i]->response);
        out += sizeof(*re);
    }

    /* Serialize phase headers */
    for (int i=0; i<map->nphases; i++) {
        lfr_phase_header_t *ph = (lfr_phase_header_t *)out;
        /* Check salt hint */
        if (i > 0 && map->phases[i]->salt != fmix64(map->phases[i-1]->salt ^ map->phases[i]->_salt_hint)) {
            return EINVAL;
        }
        if (i > 0) {
            ph->salt_hint = map->phases[i]->_salt_hint;
        } else {
            ph->salt_hint = (uint8_t) map->phases[i]->salt; // least sig byte goes in salt hint
        }
        ret = ui2le(ph->nblocks,sizeof(ph->nblocks),map->phases[i]->blocks);
        if (ret) return ret;
        out += sizeof(*ph);
    }

    /* Serialize the actual data */
    for (int i=0; i<map->nphases; i++) {
        size_t s = _lfr_uniform_map_vector_size(map->phases[i]);
        memcpy(out,map->phases[i]->data,s);
        out += s;
    }

    return 0;
}

int API_VIS lfr_nonuniform_map_deserialize(
    lfr_nonuniform_map_t map,
    const uint8_t *data,
    size_t data_size,
    uint8_t flags
) {
    memset(map,0,sizeof(map[0]));
    int ret = EINVAL;
    if (data_size < sizeof(lfr_nonuniform_header_t)) goto inval;
    const lfr_nonuniform_header_t *header = (const lfr_nonuniform_header_t*) data;

    lfr_locator_t plan = map->plan = le2ui(header->plan, sizeof(header->plan)) << LFR_INTERVAL_SH;
    map->nphases = popcount(plan);
    map->nresponses = le2ui(header->nitems, sizeof(header->nitems));
    if (map->nresponses == 0) goto inval;
   

    data += sizeof(*header);
    assert(data_size >= sizeof(*header));
    data_size -= sizeof(*header);

    if (data_size < map->nphases*(uint64_t)sizeof(lfr_phase_header_t)
               + map->nresponses*(uint64_t)sizeof(lfr_response_header_t)) {
        goto inval;
    }

    /* deserialize the response data */
    map->response_map = calloc(map->nresponses, sizeof(*map->response_map));
    if (map->response_map == NULL) goto nomem;
    unsigned seen_balance = 0; // 1+index of the 0xFF "balance of the interval" position
    lfr_locator_t base = 0;
    for (int i=0; i<map->nresponses; i++) {
        const lfr_response_header_t *re = (const lfr_response_header_t *)data;
        map->response_map[i]->response = le2ui(re->response,sizeof(re->response));
        uint8_t lg_weight = re->lg_weight;

        /* All of the intervals must be a power of 2, except at most one of them
         * (the "balance" position)
         */
        if (lg_weight == 0xFF) {
            if (seen_balance) goto inval;
            seen_balance = i+1;
            map->response_map[i]->lower_bound = base;
        } else if (lg_weight >= sizeof(lfr_locator_t)*8) {
            goto inval;
        } else {
            lfr_locator_t tmp = map->response_map[i]->lower_bound = base;
            base += (lfr_locator_t)1 << lg_weight;
            if (base < tmp && (seen_balance || i<map->nresponses-1 || base != 0)) {
                /* It wrapped around! */
                goto inval;
            }
        }

        data += sizeof(*re);
        assert(data_size >= sizeof(*re));
        data_size -= sizeof(*re);
    }
    if (seen_balance) {
        if (base == 0 && map->nresponses != 1) goto inval;
        for (int i=seen_balance; i<map->nresponses; i++) {
            map->response_map[i]->lower_bound -= base; // i.e. += balance
        }
    } else if (base != 0) {
        goto inval;
    }

    /* deserialize the phase info.  starts destroying plan */
    size_t remaining_data_required = 0;
    int phlo = ctz(map->plan);
    plan &= plan-1;
    map->phases = calloc(map->nphases, sizeof(*map->phases));
    if (map->phases == NULL) goto nomem;
    for (int i=0; i<map->nphases; i++) {
        const lfr_phase_header_t *ph = (const lfr_phase_header_t *)data;

        map->phases[i]->blocks = le2ui(ph->nblocks, sizeof(ph->nblocks));
        if (i>0) {
            map->phases[i]->_salt_hint = ph->salt_hint;
            map->phases[i]->salt = fmix64(map->phases[i-1]->salt ^ map->phases[i]->_salt_hint);
        } else {
            map->phases[i]->salt = le2ui(header->file_salt, sizeof(header->file_salt)) << 8
                                 | ph->salt_hint;
        }

        uint8_t value_bits = map->phases[i]->value_bits = (plan ? ctz(plan) : 8*sizeof(plan)) - phlo;
        phlo += value_bits;
        plan &= (plan-1);
        size_t ph_sz = (size_t)map->phases[i]->blocks * _lfr_blocksize * value_bits;
        remaining_data_required += ph_sz;
        if (remaining_data_required < ph_sz) goto inval; // overflow

        data += sizeof(*ph);
        assert(data_size >= sizeof(*ph));
        data_size -= sizeof(*ph);
    }

    if (remaining_data_required != data_size) goto inval;
    for (int i=0; i<map->nphases; i++) {
        size_t ph_sz = (size_t)map->phases[i]->blocks * _lfr_blocksize * map->phases[i]->value_bits;
        assert(data_size >= ph_sz);

        if (flags & LFR_NO_COPY_DATA) {
            map->phases[i]->data_is_mine = 0; // already
            map->phases[i]->data = data;
        } else {
            uint8_t *ph_data = malloc(ph_sz);
            if (ph_data == NULL) goto nomem;
            memcpy(ph_data, data, ph_sz);
            map->phases[i]->data = (const uint8_t *)ph_data;
            map->phases[i]->data_is_mine = 1;
        }

        data += ph_sz;
        data_size -= ph_sz;
    }
    assert(data_size == 0);

    return 0;

nomem:
    ret = ENOMEM;
    goto error;
inval:
    ret = EINVAL;
error:
    lfr_nonuniform_map_destroy(map);
    return ret;
}
