/**
 * @file lfr_builder.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 * Hash tables for building maps.
 */
#ifndef __LFR_BUILDER_H__
#define __LFR_BUILDER_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A salt value, since the hashes we use are salted. */
typedef uint64_t lfr_salt_t;

/** Response to a query */
typedef uint64_t lfr_response_t;

/** A key->value relation, used to build the maps */
typedef struct {
    const uint8_t *key;
    size_t keybytes;
    lfr_response_t value;
} lfr_relation_t;

/** A builder to store the state of a uniform map before compiling it. */
typedef struct {
    size_t used, capacity;
    size_t data_used, data_capacity;
    int copy_data;
    lfr_relation_t *relations;
    uint8_t *data;
} lfr_builder_s, lfr_builder_t[1];

/**
 * Initialize a map-builder, which may be used for either uniform or
 * non-uniform maps.  The builder stores key->value mappings, and has
 * an initial capacity of relations_capacity relations with a total
 * amount of data equal to data_capacity.
 *
 * If copy_data is nonzero, then lfr_builder_insert will copy all data
 * into a byte buffer stored in the builder.  Otherwise, data_capacity
 * may (and should) be 0.  In this case, the builder will not copy
 * or store the query's keys; they must be held externally until
 * building the map is complete.
 *
 * @param builder The builder to initialize.
 * @param relations_capacity The number of relations to allocate space for.
 * It can be increased later.
 * @param copy_data If nonzero, allocate a buffer to copy the query data.
 * @param data_capacity If copy_data is nonzero, the initial size of the
 * buffer to allocate.  It can be increased later.  If copy_data is zero,
 * this is ignored.
 * @return 0 on success.
 * @return -ENOMEM if the requested data cannot be allocated.
 */
int lfr_builder_init (
    lfr_builder_t builder,
    size_t relations_capacity,
    int copy_data,
    size_t data_capacity
);

/** Clear any relations in the map. */
void lfr_builder_reset(lfr_builder_t builder);

/** Destroy the map and free any memory it allocated. */
void lfr_builder_destroy(lfr_builder_t builder);

#ifdef __cplusplus
}; // extern "C"
#endif

#endif /* __LFR_BUILDER_H__ */
