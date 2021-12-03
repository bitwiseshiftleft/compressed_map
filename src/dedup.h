/** @file dedup.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Code to deduplicate inputs.  Used only by the higher-level
 * nonuniform maps.
 */
#ifndef __LFR_DEDUP_H__
#define __LFR_DEDUP_H__

#include <stdint.h>
#include "lfr_nonuniform.h"
#include "bitset.h"

/** Find duplicate items and mark them to be ignored.
 *
 * Each object will appear only once in the output.
 * If the same object is given a response of yes and
 * of no, then it will be treated as "yes" iff
 * yes_overrides_no is nonzero.
 * 
 * @param [inout] relevant A bitmap of relevant relations.  On return, the irrelevant duplicates will be cleared.
 * @param [out] n_out The number of irrelevant duplicates found
 * @param [in] rel The relations
 * @param [in] nrel Number of relations
 * @param yes_overrides_no If positive, if an object
 * has conflicting statuses, it ends up as yes.  If negative,
 * it ends up as no.  If zero, an error occurs.
 */
int remove_duplicates (
    bitset_t relevant,
    size_t *n_out,
    const lfr_relation_t *rel,
    size_t nrel,
    int yes_overrides_no
);
    
#endif // __LFR_DEDUP_H__
