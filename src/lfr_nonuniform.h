/**
 * @file lfr_nonuniform.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Non-uniform static functions.  These are more space-efficient
 * than uniform static functions if the distribution of values
 * differs significantly from uniform.  They should be encodable
 * to at most the Shannon entropy of the distribution, plus 11%,
 * plus small amounts of padding and metadata.
 * 
 * Note well! This is a research-grade library, and not ready for
 * production use.  Also, note that this library is not designed
 * to store secret data.  In particular, it doesn't employ side-
 * channel countermeasures, doesn't pin data to RAM, and doesn't
 * erase it with memset_s or the like.
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
    lfr_locator_t  lower_bound;
    lfr_response_t response;
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
 * @return 0 on success.
 * @return ENOMEM if we ran out of memory.
 * @return EAGAIN if we tried and failed too many times.
 *
 * @todo This function assumes a dense encoding of the responses, e.g. they
 * are 0 .. 5.  It will take huge amounts of memory if one of the responses
 * is large.
 */
int lfr_nonuniform_build (
    lfr_nonuniform_map_t map,
    const lfr_builder_t builder
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
} // extern "C"

namespace LibFrayed {
    /** Wrapper for map */
    class NonuniformMap {
    public:
        /** Wrapped map object */
        lfr_nonuniform_map_t map;

        /** Empty constructor */
        inline NonuniformMap() { memset(map,0,sizeof(map)); }

        /** Move constructor */
        inline NonuniformMap(LibFrayed::NonuniformMap &&other) {
            map[0] = other.map[0];
            memset(other.map,0,sizeof(other.map));
        }

        /** Move assignment */
        inline NonuniformMap& operator=(LibFrayed::NonuniformMap &&other) {
            lfr_nonuniform_map_destroy(map);
            map[0] = other.map[0];
            memset(other.map,0,sizeof(other.map));
            return *this;
        }

        /** Construct from a builder */
        inline NonuniformMap(const LibFrayed::Builder &builder, int nthreads=0) {
            (void)nthreads; // TODO
            int ret = lfr_nonuniform_build(map,builder.builder);
            if (ret == ENOMEM) {
                throw std::bad_alloc();
            } else if (ret == EAGAIN) {
                throw BuildFailedException();
            }
        }

        /** Destructor */
        inline ~NonuniformMap() { lfr_nonuniform_map_destroy(map); }

        /** Lookup */
        inline lfr_response_t lookup(const uint8_t *data, size_t size) const {
            return lfr_nonuniform_query(map,data,size);
        }

        /** Lookup */
        inline lfr_response_t lookup(const std::vector<uint8_t> &v) const {
            return lookup(v.data(),v.size());
        }

        /** Lookup */
        inline lfr_response_t operator[] (const std::vector<uint8_t> &v) const {
            return lookup(v);
        }
    };
}
#endif /* __cplusplus */

#endif // __LFR_NONUNIFORM_H__
