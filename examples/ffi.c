/*
 * @file ffi.c
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Tests / examples for C FFI.
 */

#include "compressed_map.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

double now() { 
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1.0e-6;
}

void test_u64_u64(uint64_t *keys, uint64_t *values, int nitems) {
    double start, end;
    printf("********** Map u64 -> u64 **********\n");

    // Build hash map
    HashMap_u64__u64 *hash = cmap_hashmap_u64_u64_new();
    start = now();
    for (int i=0; i<nitems; i++) {
        cmap_hashmap_u64_u64_insert(hash,keys[i],values[i]);
    }
    end = now();
    size_t hashmap_len = cmap_hashmap_u64_u64_len(hash);
    assert(hashmap_len <= nitems);
    assert(hashmap_len >= nitems * 0.9);
    printf("Build hashmap of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test hash map lookup, and build the values array
    // (don't build it on the first pass because values might be overwritten for duplicate keys)
    start = now();
    for (int i=0; i<nitems; i++) {
        bool ret = cmap_hashmap_u64_u64_get(hash,keys[i],&values[i]);
        (void)ret;
        assert(ret);
    }
    end = now();
    printf("Lookup in hashmap of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test compressed map build
    start = now();
    CompressedRandomMap_u64__u64 *map = cmap_compressed_random_map_u64_u64_build(hash);
    end = now();
    printf("Build compressed_random_map of %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    size_t ser_size = cmap_compressed_random_map_u64_u64_encode(map,NULL,0);
    printf("Size is %lld bytes = %0.1f bytes/item\n",
        (long long)ser_size, (double)ser_size/hashmap_len);

    // Test compressed map queries
    start = now();
    for (int i=0; i<nitems; i++) {
        uint64_t ret = cmap_compressed_random_map_u64_u64_query(map, keys[i]);
        if (ret != values[i]) {
            printf("Fail: lookup[0x%llx] should have been 0x%llx; got 0x%llx\n",
                (long long)keys[i], (long long)values[i], (long long)ret);
            exit(1);
        }
    }
    end = now();
    printf("Query compressed_random_map of %lld items: %0.3fs = %0.1f ns/item\n\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    
    cmap_hashmap_u64_u64_free(hash);
    cmap_compressed_random_map_u64_u64_free(map);

    // Build hashset
    HashSet_u64 *hashset = cmap_hashset_u64_new();
    start = now();
    for (int i=0; i<nitems; i++) {
        cmap_hashset_u64_insert(hashset,keys[i]);
    }
    end = now();
    hashmap_len = cmap_hashset_u64_len(hashset);
    assert(hashmap_len <= nitems);
    assert(hashmap_len >= nitems * 0.9);
    printf("Build hashset of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test approx set build
    start = now();
    ApproxSet_u64 *aset = cmap_approxset_u64_build(hashset,8);
    end = now();
    printf("Build approxset of %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    ser_size = cmap_approxset_u64_encode(aset,NULL,0);
    printf("Size is %lld bytes = %0.1f bytes/item\n",
        (long long)ser_size, (double)ser_size/hashmap_len);

    // Test approxset queries
    start = now();
    for (int i=0; i<nitems; i++) {
        bool ret = cmap_approxset_u64_probably_contains(aset, keys[i]);
        if (!ret) {
            printf("Fail: approxset[0x%llx] should have been true\n", (long long)keys[i]);
            exit(1);
        }
    }
    end = now();
    printf("Query approxset of %lld items: %0.3fs = %0.1f ns/item\n\n\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);


    cmap_hashset_u64_free(hashset);
    cmap_approxset_u64_free(aset);
}

void test_bytes_u64(uint64_t *keys, uint64_t *values, int nitems) {
    double start, end;
    printf("********** Map bytes -> u64 **********\n");

    // Build hash map
    HashMap_Bytes__u64 *hash = cmap_hashmap_bytes_u64_new();
    start = now();
    for (int i=0; i<nitems; i++) {
        cmap_hashmap_bytes_u64_insert(hash,(uint8_t*)&keys[i],sizeof(keys[i]),values[i]);
    }
    end = now();
    size_t hashmap_len = cmap_hashmap_bytes_u64_len(hash);
    assert(hashmap_len <= nitems);
    assert(hashmap_len >= nitems * 0.9);
    printf("Build hashmap of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test hash map lookup, and build the values array
    // (don't build it on the first pass because values might be overwritten for duplicate keys)
    start = now();
    for (int i=0; i<nitems; i++) {
        bool ret = cmap_hashmap_bytes_u64_get(hash,(uint8_t*)&keys[i],sizeof(keys[i]),&values[i]);
        (void)ret;
        assert(ret);
    }
    end = now();
    printf("Lookup in hashmap of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test compressed map build
    start = now();
    CompressedRandomMap_Bytes__u64 *map = cmap_compressed_random_map_bytes_u64_build(hash);
    end = now();
    printf("Build compressed_random_map of %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    size_t ser_size = cmap_compressed_random_map_bytes_u64_encode(map,NULL,0);
    printf("Size is %lld bytes = %0.1f bytes/item\n",
        (long long)ser_size, (double)ser_size/hashmap_len);

    // Test compressed map queries
    start = now();
    for (int i=0; i<nitems; i++) {
        uint64_t ret = cmap_compressed_random_map_bytes_u64_query(map, (uint8_t*)&keys[i], sizeof(keys[i]));
        if (ret != values[i]) {
            printf("Fail: lookup[0x%llx] should have been 0x%llx; got 0x%llx\n",
                (long long)keys[i], (long long)values[i], (long long)ret);
            exit(1);
        }
    }
    end = now();
    printf("Query compressed_random_map of %lld items: %0.3fs = %0.1f ns/item\n\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    
    cmap_hashmap_bytes_u64_free(hash);
    cmap_compressed_random_map_bytes_u64_free(map);

    // Build hashset
    HashSet_Bytes *hashset = cmap_hashset_bytes_new();
    start = now();
    for (int i=0; i<nitems; i++) {
        cmap_hashset_bytes_insert(hashset,(uint8_t*)&keys[i],sizeof(keys[i]));
    }
    end = now();
    hashmap_len = cmap_hashset_bytes_len(hashset);
    assert(hashmap_len <= nitems);
    assert(hashmap_len >= nitems * 0.9);
    printf("Build hashset of %lld / %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, (long long)nitems, end-start, (end-start)*1e9/nitems);

    // Test approx set build
    start = now();
    ApproxSet_Bytes *aset = cmap_approxset_bytes_build(hashset,8);
    end = now();
    printf("Build approxset of %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    ser_size = cmap_approxset_bytes_encode(aset,NULL,0);
    printf("Size is %lld bytes = %0.1f bytes/item\n",
        (long long)ser_size, (double)ser_size/hashmap_len);

    // Test approxset queries
    start = now();
    for (int i=0; i<nitems; i++) {
        bool ret = cmap_approxset_bytes_probably_contains(aset, (uint8_t*)&keys[i], sizeof(keys[i]));
        if (!ret) {
            printf("Fail: approxset[0x%llx] should have been true\n", (long long)keys[i]);
            exit(1);
        }
    }
    end = now();
    printf("Query approxset of %lld items: %0.3fs = %0.1f ns/item\n\n\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);


    cmap_hashset_bytes_free(hashset);
    cmap_approxset_bytes_free(aset);
}


int main(int argc, char **argv) {
    (void)argv[argc];

    int nitems = 1000000;
    if (argc > 1) {
        nitems = atoll(argv[1]);
    }

    uint64_t *keys   = calloc(nitems, sizeof(*keys));
    uint64_t *values = calloc(nitems, sizeof(*values));

    double start, end;
    srandom(0);
    start = now();
    for (int i=0; i<nitems; i++) {
        keys[i] = random() ^ (uint64_t)random()<<21 ^ (uint64_t)random()<<42;
        values[i] = random();
    }
    end = now();
    printf("Randomize %lld items: %0.3fs = %0.1f ns/item\n\n",
        (long long)nitems, end-start, (end-start)*1e9/nitems);

    test_u64_u64(keys,values,nitems);
    test_bytes_u64(keys,values,nitems);
    
    return 0;
}
