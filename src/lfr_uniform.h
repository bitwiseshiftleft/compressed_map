/**
 * @file lfr_uniform.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Uniform static functions.  These objects nearly-optimally
 * encode a map of type keys -> values, where the values are
 * iid uniform values of a certain bit length.  For simplicity,
 * we support bit lengths <= 64 here; if you want to go larger
 * you probably want a perfect hashing scheme anyway.
 *
 * Building a static function may fail, with a few % probability.
 * If it fails due to EAGAIN, you may want to retry with a different
 * salt.  Construction fails with duplicate keys even if the values
 * are the same (i.e. if you created the builder using LFR_NO_HASHTABLE
 * and then inserted a duplicate).
 * 
 * Note well! This is a research-grade library, and not ready for
 * production use.  Also, note that this library is not designed
 * to store secret data.  In particular, it doesn't employ side-
 * channel countermeasures, doesn't pin data to RAM, and doesn't
 * erase it with memset_s or the like.
 */
#ifndef __LFR_UNIFORM_H__
#define __LFR_UNIFORM_H__

#include "lfr_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************
 *                     Compiled uniform maps                     *
 *****************************************************************/

/** A compiled uniform map. */ 
typedef struct {
    size_t blocks;
    lfr_salt_t salt;
    uint8_t value_bits;
    uint8_t data_is_mine; // vector memory was allocated here, and should be deallocated with lfr_uniform_map_destroy
    uint8_t _salt_hint; // used when the salt is derived
    uint8_t *data;
} lfr_uniform_map_s, lfr_uniform_map_t[1];

/** High-level build function: using the builder, compile to a map object.
 *
 * @param map The map object.  On success, this function will initialize
 * the map and allocate memory for it.
 * @param builder The builder object.
 * @param value_bits The number of bits of the responses to use.
 * @param salt The salt to be used.
 * @return 0 on success.
 * @return ENOMEM Not enough memory to solve / return the map.
 * @return EAGAIN The solution failed; either it has inconsistent values
 * or should be tried again with a different salt.
 */
int lfr_uniform_build(lfr_uniform_map_t map, const lfr_builder_t builder, unsigned value_bits, lfr_salt_t salt);

/** As lfr_uniform_build, but if the library was built with thread support, you
 * can set the number of threads.  Set to 0 for default.  If the library was not
 * built with thread support (by default it is not), then this call ignores
 * nthreads and always uses 1 thread.
 */
int lfr_uniform_build_threaded(lfr_uniform_map_t map, const lfr_builder_t builder, unsigned value_bits, lfr_salt_t salt, int nthreads);

/** Destroy a map object, and deallocate any memory used to create it. */
void lfr_uniform_map_destroy(lfr_uniform_map_t map);

/** Return the number of bytes in the uniform map's data section. */
size_t lfr_uniform_map_size(const lfr_uniform_map_t map);

/** Query a uniform map.  If the key was used when building
 * the map, then the same value will be returned.
 */
lfr_response_t lfr_uniform_query (
    const lfr_uniform_map_t map,
    const uint8_t *key,
    size_t keybytes
);

/*****************************************************************
 *                     Testing and debugging                     *
 *****************************************************************/

/**
 * Return the number of columns required for the given number of rows.
 * It will always be a multiple of 8*LFR_BLOCKSIZE.  Useful for sizing
 * the map.  The number of bytes required for the map's data will be
 * (columns * value_bits) / 8.
 */
size_t _lfr_uniform_provision_columns(size_t rows);

/**
 * For testing purposes.  Return the maximum number of rows such that
 * _lfr_uniform_provision_columns(rows) <= cols.  For a given number of
 * columns, the failure probability increases according to the number of
 * rows, so this is the best-case scenario in terms of compression
 * efficiency but the worst-case scenario in terms of failure probability
 * and thus speed.
 */
size_t _lfr_uniform_provision_max_rows(size_t cols);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif // __LFR_UNIFORM_H__
