/**
 * @file lfr_builder.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 * Hash tables for building maps.
 */

 #include "lfr_builder.h"
 #include "util.h"
 #include <errno.h>

void API_VIS lfr_builder_destroy(lfr_builder_t builder) {
    free(builder->relations);
    free(builder->data);
    memset(builder,0,sizeof(*builder));
}

void API_VIS lfr_builder_reset(lfr_builder_t builder) {
    builder->used = 0;
    builder->data_used = 0;
}

int API_VIS lfr_builder_init (
    lfr_builder_t builder,
    size_t capacity,
    int copy_data,
    size_t data_capacity
) {
    builder->capacity = capacity;
    builder->used = 0;
    builder->data_used = 0;
    builder->copy_data = copy_data = !!copy_data;

    builder->relations = calloc(capacity, sizeof(*builder->relations));   
    if (copy_data) {
        builder->data = malloc(data_capacity); 
    } else {
        builder->data = NULL;
    }
    if (builder->relations == NULL || (copy_data && builder->data == NULL)) {
        lfr_builder_destroy(builder);
        return -ENOMEM;
    }
    
    return 0;
}

/** TODO: lfr_builder_lookup; make this return EEXIST if exists with different value */
int API_VIS lfr_builder_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    uint64_t value
) {
    const size_t CAPACITY_STEP = 1024, DATA_CAPACITY_STEP = 1<<16;
    if (builder->used >= builder->capacity) {
        size_t new_capacity = 2*builder->capacity + CAPACITY_STEP;
        lfr_relation_t *new = realloc(builder->relations, new_capacity * sizeof(*new));
        if (new == NULL) return -ENOMEM;
        builder->relations = new;
        builder->capacity = new_capacity;
    }

    if (builder->copy_data) {
        if (keybytes + builder->data_used < keybytes) return -ENOMEM; // overflow


        if (keybytes + builder->data_used > builder->data_capacity) {
            // Reallocate
            size_t new_capacity = 2*builder->data_capacity + keybytes + DATA_CAPACITY_STEP;
            if (new_capacity < keybytes + builder->data_used) return -ENOMEM; // overflow
            uint8_t *new = realloc(builder->data, new_capacity);
            if (new == NULL) return -ENOMEM;
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

    size_t row = builder->used++;
    builder->relations[row].key = key;
    builder->relations[row].keybytes = keybytes;
    builder->relations[row].value = value;
    return 0;
}
