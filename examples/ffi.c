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

int main(int argc, char **argv) {
    (void)argv[argc];

    int nitems = 1000000;
    if (argc > 1) {
        nitems = atoll(argv[1]);
    }

    uint64_t *keys   = calloc(nitems, sizeof(*keys));
    uint64_t *values = calloc(nitems, sizeof(*values));

    // Build hash map
    srandom(0);
    HashMap_u64__u64 *hash = cmap_hashmap_u64_u64_new();
    double start = now();
    for (int i=0; i<nitems; i++) {
        keys[i] = random();
        cmap_hashmap_u64_u64_insert(hash,keys[i],random());
    }
    double end = now();
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
            return 1;
        }
    }
    end = now();
    printf("Query compressed_random_map of %lld items: %0.3fs = %0.1f ns/item\n",
        (long long)hashmap_len, end-start, (end-start)*1e9/hashmap_len);
    
    cmap_hashmap_u64_u64_free(hash);
    cmap_compressed_random_map_u64_u64_free(map);
    
    return 0;
}
