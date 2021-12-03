/** @file test_lfr_nonuniform.c
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 * @brief Test and bench nonuniform maps.
 */
 #include "lfr_nonuniform.h"
#include <stdlib.h>
#include <stdio.h>
#include <sodium.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

static double now() {
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) return 0;
    return tv.tv_sec + (double)tv.tv_usec / 1e6;
}

int main(int argc, char **argv) {
    int yes_overrides_no = 1;
    unsigned nitems = 2;
    int ret;
    size_t query_length = 32;

    srandom(0);
    
    if (argc > 2) nitems = argc - 1;
    if (nitems < 1) {
        printf("nitems needs to be at least 2\n");
        return 1;
    }
    unsigned long long neach[nitems], total=0;
    for (unsigned i=0; i<nitems; i++) {
        if (argc > (int)i+1) {
            neach[i] = atoll(argv[i+1]);
        } else {
            neach[i] = 1000;
        }
        total += neach[i];
    }

    uint8_t *inputs;
     
    lfr_relation_t *relns = malloc(total * sizeof(*relns));
    
    unsigned long long neach_remaining[nitems];
    for (unsigned i=0; i<nitems; i++) { neach_remaining[i] = neach[i]; }
    
    /* Calculate entropy */
    double entropy = 0;
    for (unsigned i=0; i<nitems; i++) {
        if (neach[i]) entropy += neach[i] * log((double)neach[i] / total);
    }
    entropy /= 8*log(0.5);

    printf("Creating objects...\n");
    double start = now(), elapsed;
    inputs = malloc(query_length * total);
    for (size_t b=0; b<query_length * total; b++) {
        inputs[b] = random();
    }
    for (unsigned i=0; i<total; i++) {
        relns[i].query = &inputs[query_length*i];
        relns[i].query_length = query_length;
        
        size_t it = (random() ^ (size_t)random()<<20) % (total-i);
        relns[i].response = 0;
        for (unsigned j=0; j<nitems; j++) {
            if (it < neach_remaining[j]) {
                relns[i].response = j;
                neach_remaining[j]--;
                break;
            } else {
                it -= neach_remaining[j];
            }
        }
    }

    
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.1f usec/call\n",
        elapsed, elapsed * 1e6 / total);
    
    lfr_nonuniform_map_t map;
        
    printf("Create lfr_nonuniform...\n");
    start = now();
    ret = lfr_nonuniform_build(map, relns, total, yes_overrides_no);
    if (ret) {
        printf("Sketch failed!\n");
        return -1;
    }
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.3f usec/row\n", elapsed, elapsed * 1e6 / total);
    
    printf("Sketch created; sanity checking...\n");
    start = now();
    for (size_t i=0; i<total; i++) {
        lfr_response_t answer = lfr_nonuniform_query(map,relns[i].query,relns[i].query_length);
        if (answer != relns[i].response) {
            printf("Bug: query %lld answer should be %d but query gave %d\n",
                (unsigned long long)i, (int)relns[i].response, (int)answer);
        }
    }
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.3f usec/query\n",
        elapsed, elapsed * 1e6 / total
    );
    
    size_t size = 0;
    for (int i=0; i<map->nphases; i++) {
        size += lfr_uniform_map_size(map->phases[i]);
    }
    double ratio = entropy ? size / entropy : INFINITY;
    printf("size = %lld bytes, shannon = %d bytes, ratio = %0.3f\n", (long long) size, (int)entropy, ratio);
    
    lfr_nonuniform_map_destroy(map);
    return 0;
}
