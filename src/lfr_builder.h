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

#define LFR_NO_COPY_DATA (1<<0) /** Don't copy the data; caller must hold it. */
#define LFR_NO_HASHTABLE (1<<1) /** Don't hash to dedup; caller is responsible for dedup. */

/** A builder to store the state of a uniform map before compiling it. */
typedef struct {
    size_t used, capacity;
    size_t data_used, data_capacity;
    size_t hash_capacity;
    lfr_salt_t salt;
    lfr_relation_t *relations;
    lfr_relation_t **hashtable;
    uint8_t *data;
    uint8_t flags;
} lfr_builder_s, lfr_builder_t[1];

/**
 * Initialize a map-builder, which may be used for either uniform or
 * non-uniform maps.  The builder stores key->value mappings, and has
 * an initial capacity of relations_capacity relations with a total
 * amount of data equal to data_capacity.
 *
 * If (flags & LFR_NO_COPY_DATA)==0, then lfr_builder_insert will copy
 * all data into a byte buffer stored in the builder.  Otherwise, the
 * keys must be held externally until building the map is complete.
 *
 * @param builder The builder to initialize.
 * @param relations_capacity The number of relations to allocate space for.
 * It can be increased later.
 * @param data_capacity Initial size of the buffer to allocate.  Ignored if
 * (flags & LFR_NO_COPY_DATA).
 * @param flags Flags to the object.
 * @return 0 on success.
 * @return -ENOMEM if the requested data cannot be allocated.
 */
int lfr_builder_init (
    lfr_builder_t builder,
    size_t relations_capacity,
    size_t data_capacity,
    uint8_t flags
);

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
 * @return -ENOMEM if the map is over the allocated capacity,
 * and the size cannot be increased.
 * @return -EEXIST if the key already exists in the map, with
 * a different value.
 */
int lfr_builder_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    lfr_response_t value
);

/**
 * Lookup a key in the builder's hashtable.  If the item isn't
 * found, or if the builder was created with LFR_NO_HASHTABLE flag,
 * then return NULL.
 * @param builder The uniform map-builder object.
 * @param key Pointer to the key data.
 * @param keybytes Length of the key data in bytes.
 */
lfr_response_t *lfr_builder_lookup (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes
);

/**
 * Lookup a key in the builder's hashtable.  If the item isn't
 * found, or if the builder was created with the LFR_NO_HASHTABLE
 * flag, insert value_if_not_found.
 * @param builder The uniform map-builder object.
 * @param key Pointer to the key data.
 * @param keybytes Length of the key data in bytes.
 * @param value_if_not_found Length of the key data in bytes.
 * @return A pointer to the value.
 * @return NULL if out of memory.
 */
lfr_response_t *lfr_builder_lookup_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    lfr_response_t value_if_not_found
);

/** Clear any relations in the map. */
void lfr_builder_reset(lfr_builder_t builder);

/** Destroy the map and free any memory it allocated. */
void lfr_builder_destroy(lfr_builder_t builder);

#ifdef __cplusplus
}; /* extern "C" */
#endif /* __cplusplus */

#endif /* __LFR_BUILDER_H__ */
