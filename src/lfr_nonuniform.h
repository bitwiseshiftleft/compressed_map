/**
 * @file lfr_nonuniform.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Non-uniform static functions.  These are more space-efficient
 * than uniform static functions if the distribution of values
 * differs significantly from uniform.
 *
 * They alse differ from the uniform case with respect to retries.
 * In the nonuniform case, if building fails, it simply returns -EAGAIN
 * But the nonuniform case proceeds in several phases, each of which
 * requires building an lfr_uniform structure.  This would amplify the
 * probability of failure.  Therefore the nonuniform case stores the
 * entire input, so that it can re-choose salts and retry individual
 * phases if they fail.
 */
#ifndef __LFR_NONUNIFORM_H__
#define __LFR_NONUNIFORM_H__

#include <stddef.h>
#include <stdint.h>
#include "lfr_uniform.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Locator interval value */
typedef uint64_t lfr_locator_t;

/** An interval structure used in the map to store a response. */
typedef struct {
    lfr_locator_t lower_bound;
    lfr_response_t     response;
} lfr_nonuniform_intervals_s, lfr_nonuniform_intervals_t[1];

/** A nonuniform map structure, ready to be queried */
typedef struct {
    lfr_locator_t plan;
    int nresponses;
    int nphases;
    lfr_nonuniform_intervals_t *response_map;
    lfr_uniform_map_t *phases;
} lfr_nonuniform_map_s, lfr_nonuniform_map_t[1];

/**
 * Create a nonuniform static function from a collection of relations.
 * @param map The map
 * @param builder The relation data
 * @param dedup_direction If zero, if there are duplicate keys in the map
 *   with differing values, return -EEXIST.  If positive, use the largest
 *   value for those keys.  If negative, use the smallest value.
 * @return 0 on success.
 * @return -ENOMEM if we ran out of memory.
 * @return -EEXIST if duplicate keys were present.
 * @return -EAGAIN if we tried and failed too many times.
 *
 * @todo The duplicate detection is a hack.  It's slow, and it's done
 * after the map size is estimated, which means that a map with very
 * many duplicates will use bad estimates when setting its parameters.
 *
 * @todo This function assumes a dense encoding of the responses, e.g. they
 * are 0 .. 5.  It will take huge amounts of memory if one of the responses
 * is large.
 */
int lfr_nonuniform_build (
    lfr_nonuniform_map_t map,
    const lfr_builder_t builder,
    int dedup_direction
);

/** Destroy the map, deallocate its memory (except the struct) and zeroize it */
void lfr_nonuniform_map_destroy(lfr_nonuniform_map_t map);
    
/** Query a nonuniform map, with a key that's `keybytes` bytes long. */
lfr_response_t lfr_nonuniform_query (
    const lfr_nonuniform_map_t map,
    const uint8_t *key,
    size_t keybytes
);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif // __LFR_NONUNIFORM_H__
