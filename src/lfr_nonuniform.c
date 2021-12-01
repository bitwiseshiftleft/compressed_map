/** @file lfr_nonuniform.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
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
#include "util.h"
#include "dedup.h"

#define LFR_INTERVAL_BYTES (40/8)
#define LFR_INTERVAL_SH (8*(sizeof(lfr_locator_t) - LFR_INTERVAL_BYTES))

/* Outer header.  TODO: move to new serialize implementation */
#define LFR_HEADER_MY_VERSION 0
#define LFR_FILE_SALT_BYTES 4
#define LFR_BUCKET_PHASE_BYTES_BYTES  (32/8) // limit: ~32 billion items / bucket
#define LFR_BUCKET_OFFSET_BYTES_BYTES (48/8) // limit: ~256 terabytes
#define LFR_TOTAL_SIZE_BYTES (48/8)
#define LFR_NITEMS_BYTES (32/8)
typedef struct {
    uint8_t version;
    uint8_t total_size[LFR_TOTAL_SIZE_BYTES];
    uint8_t plan[LFR_INTERVAL_BYTES];
    uint8_t file_salt[LFR_FILE_SALT_BYTES];
    uint8_t nitems[LFR_NITEMS_BYTES];
} __attribute__((packed)) lfr_nonuniform_header_t;

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
    lfr_nonuniform_response_t resp; /** The response / dictionary value itself */
} formulation_item_t;

/** Sort by interval width */
static int cmp_fit(const void *va, const void *vb) {
    const formulation_item_t *a = va, *b = vb;
    if (a->width < b->width) return -1;
    if (a->width > b->width) return 1;
    if (a->count < b->count) return -1;
    if (a->count > b->count) return 1;
    if (a->resp < b->resp) return -1;
    if (a->resp > b->resp) return 1;
    return 0;
}

/** Sort by response */
static int cmp_item(const void *va, const void *vb) {
    const formulation_item_t *a = va, *b = vb;
    if (a->resp < b->resp) return -1;
    if (a->resp > b->resp) return 1;
    
    // Items with the same response must be the same item.
    assert(a->width == b->width && a->count == b->count);
    return 0;
}

/**
 * Given the counts for each value, optimize the intervals assigned to each
 * and compute a "plan" measuring where the phases begin and end.
 */
static lfr_locator_t lfr_nonuniform_formulate_plan (
    lfr_nonuniform_intervals_t *response_map,
    const size_t *item_counts,
    unsigned nitems
) {
    size_t total = 0;
    /* Compute initial assignments as a starting point for the LP */
    for (unsigned i=0; i<nitems; i++) {
        total += item_counts[i];
    }
    
    /* Sort by width ~ count, ascending. */
    size_t ratio = (size_t)(-1) / total;
    lfr_locator_t silt_1 = 1;
    formulation_item_t items[nitems]; // TODO: heap alloc?
    for (unsigned i=0; i<nitems; i++) {
        int bit = high_bit(item_counts[i] * ratio);
        if (bit < (int)LFR_INTERVAL_SH) bit = LFR_INTERVAL_SH;
        items[i].width = item_counts[i] ? silt_1 << bit : 0;
        items[i].resp  = i; // TODO: support nonsequential values
        items[i].count = item_counts[i];
    }
    qsort(items,nitems,sizeof(items[0]),cmp_fit);

    /* The interval widths need to add to 1.  But they might not because they're
     * rounded up/down to powers of 2.
     */
    total = 0;
    for (unsigned i=0; i<nitems-1; i++) {
        assert(total + items[i].width >= total); // TODO: deal with the overflow case?  Can it even happen?
        total += items[i].width;
    }
    items[nitems-1].width = -total;
    
    /* Simple linear programming to determine the optimal interval allocation.
     *
     * The derivative of expected file size wrt the assigned widths is piecewise
     * linear, and it's naturally nicely factored: moving interval width from
     * the biggest interval to any other affects only the sizes for those two
     * intervals.  So it should be quick -- TODO, formally analyze -- to find
     * an optimum just by hill climbing.
     *
     * PERF: in theory this should use a prio queue or something
     * For only a few items it doesn't matter though.
     *
     * TODO: fully verify that this works.
     */
    lfr_locator_t half = silt_1 << (sizeof(half)*8-1);
    unsigned ADJ_LIMIT = 1<<16; // TODO
    for (unsigned adj=0; adj<ADJ_LIMIT; adj++) {
        /* Calculate, for each interval, the derivative with respect to
         * increasing or decreasing it.  These are usually different, because
         * most items are at powers of 2, which is knee in the graph.
         */
        double max_up = -INFINITY;
        double min_dn = INFINITY;
        int i_max_up = 0, i_min_dn = 0;
        for (unsigned i=0; i<nitems; i++) {
            double deriv_up = -INFINITY;
            if (items[i].width < half) {
                deriv_up = ldexp(items[i].count, -high_bit(items[i].width));
            }
            if (deriv_up > max_up) {
                max_up = deriv_up;
                i_max_up = i;
            }
            
            double deriv_dn = INFINITY;
            if (items[i].width > 1) {
                deriv_dn = ldexp(items[i].count, -high_bit(items[i].width-1));
            }
            if (deriv_dn < min_dn) {
                min_dn = deriv_dn;
                i_min_dn = i;
            }
        }

        if (max_up <= min_dn) {
            /* All gradient directions are negative */
            break;
        }
        
        /* OK, we found positions where the tradeoff is favorable.  Make that tradeoff until
         * one or the other hits the next knee.
         */
        lfr_locator_t cap_up = ((lfr_locator_t)2 << high_bit(items[i_max_up].width))
            - items[i_max_up].width;
        lfr_locator_t cap_dn =
            items[i_min_dn].width - ((lfr_locator_t)1 << high_bit(items[i_min_dn].width-1));
        lfr_locator_t cap = (cap_up < cap_dn) ? cap_up : cap_dn;
        
        items[i_max_up].width += cap;
        items[i_min_dn].width -= cap;
    }

    /* Sort and calculate the interval bounds.
     * TODO: use a hash table to allow non-dense encodings.
     * TODO: when doing so, should probably sort this to naturally
     * align all intervals.
     * TODO: this may open up other optimization opportunities.
     */
    qsort(items,nitems,sizeof(items[0]),cmp_item);
    total = 0;
    for (unsigned i=0; i<nitems; i++) {
        response_map[i]->lower_bound = total;
        response_map[i]->response = i;
        total += items[i].width;
    }
    
    /* Calculate the plan, which is a bitmask of where the phases begin/end */
    return lfr_nonuniform_summarize_plan(response_map, nitems);
}

static size_t get_nconstr_target (
    lfr_locator_t plan,
    int phase,
    unsigned nitems,
    const size_t *item_counts,
    const lfr_nonuniform_intervals_t *response_map
) {
    if (!plan) return 0;
    if (!phase) return SIZE_MAX; // phase 0 is deterministic

    /* False positives must be at most FP_TIGHTNESS_MULT * expected + FP_TIGHTNESS_ADD */
    const float FP_TIGHTNESS_MULT = 1.005; 
    const int FP_TIGHTNESS_ADD = 32;

    /* All previous phases set plan at random */
    for (int i=0; i<phase; i++) plan &= plan-1;
    
    lfr_locator_t plan_interval = plan &~ (plan-1);
    
    size_t total=0;
    for (unsigned i=0; i<nitems; i++) {
        lfr_locator_t interval_width = response_map[(i+1) % nitems]->lower_bound - response_map[i]->lower_bound;
        size_t count = item_counts[i];
        if (interval_width <= plan_interval) {
            total += count;
        } else if (interval_width/2 <= plan_interval) {
            double frac = 2 - (double)interval_width / plan_interval;
            total += frac*count * FP_TIGHTNESS_MULT + FP_TIGHTNESS_ADD;
        }
    }
    return total;
}

/** Binary search for item in map */
static lfr_nonuniform_response_t bsearch_bound (
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
    if (high <= lowx && high>>phlo == lowx>>phlo) {
        /* The interval wraps, and includes all multiples of 1<<phlo */
        return 0;
    } else {
        *constraint = high>>phlo;
        return 1;
    }
}

/* TODO better failure handling */
#ifndef LFR_MAX_FAILURES
#define LFR_MAX_FAILURES 100
#endif
#ifndef LFR_PHASE_TRIES
#define LFR_PHASE_TRIES 3
#endif

/* Create a lfr_nonuniform. */
int API_VIS lfr_nonuniform_build (
    lfr_nonuniform_map_t out,
    const lfr_nonuniform_relation_t *relns,
    size_t nrelns,
    int yes_overrides_no
) { 
    /*************************************************************
     * Setup and counting phase.
     * Remove duplicates, count the items, formulate a plan,
     * allocate space, create headers.
     *************************************************************/

    /* Preinitialize so that we can goto done */
    memset(out,0,sizeof(*out));
    int ret = -1;
    size_t *item_counts = NULL;
    lfr_locator_t plan;
    lfr_locator_t *current = NULL;
    bitset_t relevant = NULL;

    int *phase_salt = NULL;
    
    // Find the maximum response
    // TODO: support nitems = 0 or 1, or non-dense encodings (using a hash table)
    unsigned MAX_NITEMS = 1<<16, nitems=2;
    for (size_t i=0; i<nrelns; i++) {
        lfr_nonuniform_response_t resp = relns[i].response;
        if (resp >= MAX_NITEMS) return -1;
        else if (resp+1 > nitems) nitems = resp+1;
    }

    item_counts = calloc(nitems,sizeof(*item_counts));
    if (item_counts == NULL) goto alloc_failed;

    /* Count the items */
    for (size_t i=0; i<nrelns; i++) {
        lfr_nonuniform_response_t resp = relns[i].response;
        assert(resp <= nitems);
        item_counts[resp]++;
    }
    
    /* Create the response map */
    out->response_map = calloc(nitems, sizeof(*out->response_map));
    if (out->response_map == NULL) goto alloc_failed;
    out->nresponses = nitems;

    /* Create plan and interval bounds */
    out->plan = plan = lfr_nonuniform_formulate_plan(out->response_map, item_counts, nitems);
    int nphases = popcount(plan);

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
    
    lfr_uniform_builder_t builder;
    memset(builder,0,sizeof(builder));

    // Search tree for suitable salts
    phase_salt[0] = 0;
    int phase=0;
    for (int try=0; phase >= 0 && phase < nphases && try < nphases + LFR_MAX_FAILURES; try++) {
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
            lfr_nonuniform_response_t resp = relns[i].response;
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

        /* Remove duplicates.  Done on phase data because it's slow */
        size_t ndupes;
        if (( ret = remove_duplicates (
            relevant, &ndupes, relns, nrelns, yes_overrides_no
        ))) { goto done; }
        nconstraints -= ndupes;
        
        if (nconstraints > get_nconstr_target(plan, phase, nitems, item_counts, out->response_map)) {
            // Too many false positives from previous phase
            phase--;
            continue;
        }
        
        /* Create the builder */
        lfr_uniform_builder_destroy(builder);

        lfr_salt_t salt;
        if (phase == 0) {
            /* (re)randomize salt for the whole operation */
            if (( ret = getentropy(&salt, sizeof(salt)) )) goto done;
        } else {
            salt = fmix64(out->phases[phase-1]->salt ^ phase_salt[phase]);
        }

        ret = lfr_uniform_builder_init(builder, nconstraints, phhi-phlo+1, salt); // no salt yet, set in iteration
        if (ret) { goto done; }

        /* Build the uniform map using constrained items */
        for (size_t i=0; i<nrelns; i++) {
            if (!bitset_test_bit(relevant, i)) continue; // it's not constrained this phase
            lfr_nonuniform_response_t resp = relns[i].response;
        
            lfr_locator_t
                lowx = out->response_map[resp]->lower_bound-1,
                high = out->response_map[(resp+1) % nitems]->lower_bound-1,
                cur = current[i],
                constraint;
                                
            int c = constrained_this_phase(&constraint, phlo, phhi, cur, lowx, high);
            assert(c);
            lfr_uniform_insert(builder, relns[i].query, relns[i].query_length, constraint);
        }
        
        lfr_uniform_map_destroy(out->phases[phase]);
        int phase_ret = lfr_uniform_build(out->phases[phase], builder);
        out->phases[phase]->_salt_hint = phase_salt[phase];
        if (phase_ret == 0 && phase < nphases-1) {
            /* It's not the last phase.  Adjust the values of all items.
             *
             * TODO PERF: Once intervals are naturally aligned, we don't need
             * to do this for most items: they take on a particular value in
             * their aligned phase, and a fixed 0/1 in all later phases
             * depending whether we start with second-most or second-least
             * common answer.
             *
             * The exception is the most common response, since the end of its
             * interval won't necessarily be naturally aligned.
             */
            for (size_t i=0; i<nrelns; i++) {
                lfr_locator_t ci = current[i], mask=((lfr_locator_t)1<<phlo)-1;
                ci &= mask;
                ci += lfr_uniform_query(out->phases[phase], relns[i].query, relns[i].query_length) << phlo;
                current[i] = ci;
            }
        }

        if (phase_ret == 0) {
            // Success!
            phase++;
            phase_salt[phase] = 0;
        }
    }
    
    if (phase < nphases) ret = -EAGAIN;
    goto done;

alloc_failed:
    ret = -ENOMEM;

done:
    /* Clean up all allocations */
    free(phase_salt);
    bitset_destroy(relevant);
    free(current);
    lfr_uniform_builder_destroy(builder);
    free(item_counts);
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

lfr_nonuniform_response_t API_VIS lfr_nonuniform_query (
    const lfr_nonuniform_map_t map,
    const uint8_t *key,
    size_t keybytes
) {
    if (map->nphases <= 0) return map->response_map[0]->response;
    lfr_locator_t loc=0, plan=map->plan, known_mask = (plan-1) &~ plan;

    /* Start with the second-last phase, because it determines the most items */
    int starting_phase = map->nphases-2;
    if (starting_phase < 0) starting_phase = 0;
    for (int i=0; i<starting_phase; i++) plan &= plan-1;

    for (int phase=starting_phase, looped=0;; phase++) {
        if (phase >= map->nphases) {
            looped = 1;
            phase = 0;
            plan = map->plan;
        }
        if (looped && phase == starting_phase) break; // shouldn't happen
        lfr_locator_t thisphase = lfr_uniform_query(map->phases[phase], key, keybytes);

        // Track what we now know
        lfr_locator_t least = plan &~ (plan-1);
        loc += thisphase * least;
        plan -= least;
        known_mask |= (plan - least) &~ plan;
        
        lfr_nonuniform_response_t lower = bsearch_bound(map->nresponses,map->response_map,loc);
        lfr_nonuniform_response_t upper = bsearch_bound(map->nresponses,map->response_map,loc |~ known_mask);
        if (upper == lower) {
            return upper;
        }
    };
    
    assert(0 && "bug or map is corrupt: lfr_nonuniform_query should have narrowed down a response");
    return -(lfr_nonuniform_response_t)1;
}
