/**
 * @file lfr_builder.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 * Hash tables for building maps.
 */

#include "lfr_builder.h"
#include "util.h"
#include <errno.h>
#include <sys/random.h>
#include <unistd.h>

static const float LFR_HASHTABLE_OVERPROVISION = 1.5;

void API_VIS lfr_builder_destroy(lfr_builder_t builder) {
    free(builder->relations);
    free(builder->data);
    free(builder->hashtable);
    memset(builder,0,sizeof(*builder));
}

void API_VIS lfr_builder_reset(lfr_builder_t builder) {
    builder->used = 0;
    builder->data_used = 0;

    /* Clear the hash table */
    for (size_t i=0; i<builder->hash_capacity; i++) {
        builder->hashtable[i] = NULL;
    }
}

int API_VIS lfr_builder_init (
    lfr_builder_t builder,
    size_t capacity,
    size_t data_capacity,
    uint8_t flags
) {
    builder->capacity = capacity;
    builder->used = 0;
    builder->data_used = 0;
    builder->hash_capacity = 0;
    builder->flags = flags;
    builder->data = NULL;
    builder->relations = NULL;
    builder->hashtable = NULL;

    /* Choose random salt */
    int ret = getentropy(&builder->salt, sizeof(builder->salt));
    if (ret) {
        lfr_builder_destroy(builder);
        return ret;
    }

    /* Allocate the relations */
    builder->relations = calloc(capacity, sizeof(*builder->relations));  
    if (builder->relations == NULL) {
        lfr_builder_destroy(builder);
        return ENOMEM;
    }

    /* Allocate the hashtable if required */
    if (!(flags & LFR_NO_HASHTABLE)) {
        size_t hash_capacity = builder->hash_capacity = capacity * LFR_HASHTABLE_OVERPROVISION;
        builder->hashtable = malloc(hash_capacity * sizeof(*builder->hashtable));
        if (builder->hashtable == NULL) {
            lfr_builder_destroy(builder);
            return ENOMEM;
        }
        for (size_t i=0; i<hash_capacity; i++) {
            builder->hashtable[i] = NULL;
        }
    }

    /* Allocate the data buffer, if required */
    if (!(flags & LFR_NO_COPY_DATA)) {
        builder->data = malloc(data_capacity);
        if (builder->data == NULL) {
            lfr_builder_destroy(builder);
            return ENOMEM;
        }
    }
    
    return 0;
}

static lfr_response_t *lfr_builder_really_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    lfr_response_t value,
    uint64_t hash
) {
    const size_t CAPACITY_STEP = 1024, DATA_CAPACITY_STEP = 1<<16;

    /* Make sure we have space */
    if (builder->used >= builder->capacity) {
        size_t new_capacity = 2*builder->capacity + CAPACITY_STEP;
        size_t hash_capacity = new_capacity * LFR_HASHTABLE_OVERPROVISION;

        if (!(builder->flags & LFR_NO_HASHTABLE)) {
            /* Expand the hash table.  Do this first: otherwise if realloc'ing the relations
                * were to fail, but this step were to succeed, then we would still have to rebuild
                * the hash table to adjust the pointers.
                */
            lfr_relation_t **newh = realloc(builder->hashtable, hash_capacity * sizeof(*builder->hashtable));
            if (newh == NULL) return NULL;
            builder->hashtable = newh;
        }

        lfr_relation_t *new = realloc(builder->relations, new_capacity * sizeof(*new));
        if (new == NULL) return NULL;
        builder->relations = new;
        builder->capacity = new_capacity;

        if (!(builder->flags & LFR_NO_HASHTABLE)) {
            /* Rebuild the hash table */
            for (size_t i=0; i<hash_capacity; i++) {
                builder->hashtable[i] = NULL;
            }
            for (size_t i=0; i<builder->used; i++) {
                uint64_t a_hash = lfr_hash(
                    builder->relations[i].key,
                    builder->relations[i].keybytes,
                    builder->salt
                ).low64;
                a_hash %= builder->hash_capacity;
                for (; builder->hashtable[a_hash] != NULL; a_hash = (a_hash+1) % builder->hash_capacity) {}
                builder->hashtable[a_hash] = &builder->relations[i];
            }

            /* Refind the insertion point */
            hash = lfr_hash(key,keybytes,builder->salt).low64;
            hash %= builder->hash_capacity;
            for (; builder->hashtable[hash] != NULL; hash = (hash+1) % builder->hash_capacity) {}

            builder->hash_capacity = hash_capacity;
        }
    }

    /* Copy the data if applicable */
    if (!(builder->flags & LFR_NO_COPY_DATA)) {
        if (keybytes + builder->data_used < keybytes) return NULL; // overflow
        if (keybytes + builder->data_used > builder->data_capacity) {
            // Reallocate
            size_t new_capacity = 2*builder->data_capacity + keybytes + DATA_CAPACITY_STEP;
            if (new_capacity < keybytes + builder->data_used) return NULL; // overflow
            uint8_t *new = realloc(builder->data, new_capacity);
            if (new == NULL) return NULL;

            // patch up the pointers
            for (size_t i=0; i<builder->used; i++) {
                builder->relations[i].key = new + (builder->relations[i].key - builder->data);
            }
            builder->data = new;
            builder->data_capacity = new_capacity;
        }

        assert(keybytes + builder->data_used <= builder->data_capacity);
        memcpy(&builder->data[builder->data_used], key, keybytes);
        key = &builder->data[builder->data_used];
        builder->data_used += keybytes;
    }

    /* Create the row */
    size_t row = builder->used++;
    builder->relations[row].key = key;
    builder->relations[row].keybytes = keybytes;
    builder->relations[row].value = value;

    if (!(builder->flags & LFR_NO_HASHTABLE)) {
        /* Insert also in hashtable */
        builder->hashtable[hash] = &builder->relations[row];
    }
    return &builder->relations[row].value;
}

static lfr_response_t *lfr_builder_lookup_core (
    const lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    uint64_t *hash_p /* Return to save time */
) {
    /* Look up in the hashtable */
    lfr_relation_t *ret = NULL;
    uint64_t hash=0;
    if (!(builder->flags & LFR_NO_HASHTABLE)) {
        hash = lfr_hash(key,keybytes,builder->salt).low64;
        hash %= builder->hash_capacity;
        for (; (ret = builder->hashtable[hash]) != NULL; hash = (hash+1) % builder->hash_capacity) {
            if (ret->keybytes == keybytes && !bcmp(ret->key,key,keybytes)) break;
        }
    }

    if (hash_p) *hash_p = hash;
    return (ret != NULL) ? &ret->value : NULL;
}

lfr_response_t *API_VIS lfr_builder_lookup (
    const lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes
) {
    return lfr_builder_lookup_core(builder,key,keybytes,NULL);
}

lfr_response_t *API_VIS lfr_builder_lookup_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    lfr_response_t value
) {
    uint64_t hash;
    lfr_response_t *found = lfr_builder_lookup_core(builder,key,keybytes,&hash);
    if (found == NULL) {
        found = lfr_builder_really_insert(builder,key,keybytes,value,hash);
    }
    return found;
}

int API_VIS lfr_builder_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    lfr_response_t value
) {
    uint64_t hash;
    lfr_response_t *found = lfr_builder_lookup_core(builder,key,keybytes,&hash);
    if (found == NULL) {
        found = lfr_builder_really_insert(builder,key,keybytes,value,hash);
        if (found == NULL) return ENOMEM;
    } else if (*found != value) {
        return EEXIST;
    }
    return 0;
}
