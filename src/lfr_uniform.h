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
 * salt.  The constructor doesn't store your keys, only the hashes,
 * so it can't retry automatically.  For the same reason, it can't
 * tell whether the construction failed due to bad luck, or because
 * you have duplicate keys.
 *
 * Construction fails with duplicate keys even if the values are
 * the same.
 * 
 * Note well! This is a research-grade library, and not ready for
 * production use.  Also, note that this library is not designed
 * to store secret data.  In particular, it doesn't employ side-
 * channel countermeasures, doesn't pin data to RAM, and doesn't
 * erase it with memset_s or the like.
 */
#ifndef __LFR_UNIFORM_H__
#define __LFR_UNIFORM_H__

#include <stdint.h>
#include <stddef.h>

#ifndef LFR_BLOCKSIZE
/* The block size in bytes.  You can change it for research purposes,
 * but the resulting library will be incompatible.
 */
#define LFR_BLOCKSIZE 4
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************
 *                     Uniform map builders                      *
 *****************************************************************/

/** A full-size salt value */
typedef uint64_t lfr_salt_t;

/** Block indices */
typedef uint32_t lfr_uniform_block_index_t;

/** Pairs of block indices **/
typedef struct { lfr_uniform_block_index_t blocks[2]; } _lfr_uniform_row_indices_s;

/** A builder to store the state of a uniform map before compiling it. */
typedef struct {
    size_t used, capacity;
    size_t blocks;
    lfr_salt_t salt;
    uint8_t value_bits;
    _lfr_uniform_row_indices_s *row_meta;
    uint8_t *data;
} lfr_uniform_builder_s, lfr_uniform_builder_t[1];


/** Initialize a map of the given capacity.  The size of the map when
 * built depends on its capacity, not how many rows are actually added.
 */
int lfr_uniform_builder_init (
    lfr_uniform_builder_t map,
    size_t capacity,
    size_t value_bits,
    lfr_salt_t salt
);

/** Clear any relations in the map. */
void lfr_uniform_builder_reset(lfr_uniform_builder_t builder);

/** Destroy the map and free any memory it allocated. */
void lfr_uniform_builder_destroy(lfr_uniform_builder_t builder);

/**
 * Set a relation in the map, so that when queried with
 * the given key, it will return a value matching the given
 * value.  Only the least-significant map->value_bits bits
 * will be stored.
 *
 * @param builder The uniform map-builder object.
 * @param key Pointer to the key data.
 * @param keybytes Length of the key data in bytes.
 * @param value The value to associate to the key.
 * @return 0 on success.
 * @return -EINVAL if the map is already at capacity.
 */
int lfr_uniform_insert (
    lfr_uniform_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    uint64_t value
);

/** Return the number of bytes required to store data section of the map once
 * it is built.
 */
size_t lfr_uniform_builder_size(const lfr_uniform_builder_t builder);


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
 * @param builder The builder object.
 * @param map The map object.  On success, this function will initialize
 * the map and allocate memory for it.
 * @return 0 on success.
 * @return -ENOMEM Not enough memory to solve / return the map.
 * @return -EAGAIN The solution failed; either it has inconsistent values
 * or should be tried again with a different seed.
 */
int lfr_uniform_build(lfr_uniform_map_t map, const lfr_uniform_builder_t builder);

/** As lfr_uniform_build, but if the library was built with thread support, you
 * can set the number of threads.  Set to 0 for default.  If the library was not
 * built with thread support (by default it is not), then this call ignores
 * nthreads and always uses 1 thread.
 */
int lfr_uniform_build_threaded(lfr_uniform_map_t map, const lfr_uniform_builder_t builder, int nthreads);

/** Destroy a map object, and deallocate any memory used to create it. */
void lfr_uniform_map_destroy(lfr_uniform_map_t map);

/** Return the number of bytes in the uniform map's data section. */
size_t lfr_uniform_map_size(const lfr_uniform_map_t map);

/** Query a uniform map.  If the key was used when building
 * the map, then the same value will be returned.
 */
uint64_t lfr_uniform_query (
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
