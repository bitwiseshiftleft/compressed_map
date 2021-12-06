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
    unsigned nitems = 2;
    size_t keybytes = 32;

    srandom(0);
    
    if (argc > 2) nitems = argc - 1;
    if (nitems < 1) {
        printf("nitems needs to be at least 2\n");
        return 1;
    }

    unsigned long long total=0;
    std::vector<unsigned long long> neach(nitems), neach_remaining(nitems);
    for (unsigned i=0; i<nitems; i++) {
        if (argc > (int)i+1) {
            neach[i] = atoll(argv[i+1]);
        } else {
            neach[i] = 1000;
        }
        total += neach[i];
        neach_remaining[i] = neach[i];
    }

    uint8_t *inputs;
    
    /* Calculate entropy */
    double entropy = 0;
    for (unsigned i=0; i<nitems; i++) {
        if (neach[i]) entropy += neach[i] * log((double)neach[i] / total);
    }
    entropy /= 8*log(0.5);

    printf("Creating objects...\n");
    double start = now(), elapsed;
    inputs = (uint8_t*)malloc(keybytes * total);
    for (size_t b=0; b<keybytes * total; b++) {
        inputs[b] = random();
    }

    LibFrayed::Builder builder(total,0,LFR_NO_COPY_DATA);
    for (unsigned i=0; i<total; i++) {
        int resp = 0;
        size_t it = (random() ^ (size_t)random()<<20) % (total-i);
        for (unsigned j=0; j<nitems; j++) {
            if (it < neach_remaining[j]) {
                resp = j;
                neach_remaining[j]--;
                break;
            } else {
                it -= neach_remaining[j];
            }
        }

        builder.lookup(&inputs[keybytes*i],keybytes) = resp;
    }

    
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.1f usec/call\n",
        elapsed, elapsed * 1e6 / total);
        
    printf("Create lfr_nonuniform...\n");
    start = now();
    LibFrayed::NonuniformMap map(builder);
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.3f usec/row\n", elapsed, elapsed * 1e6 / total);
    
    printf("Sketch created; sanity checking...\n");
    start = now();
    for (size_t i=0; i<total; i++) {
        lfr_response_t answer = map.lookup(builder[i].key,keybytes);
        if (answer != builder[i].value) {
            printf("Bug: query %lld answer should be %d but query gave %d\n",
                (unsigned long long)i, (int)builder[i].value, (int)answer);
        }
    }
    elapsed = now()-start;
    printf("   ... took %0.3f seconds = %0.3f usec/query\n",
        elapsed, elapsed * 1e6 / total
    );
    
    size_t size = 0;
    for (int i=0; i<map.map->nphases; i++) {
        size += lfr_uniform_map_size(map.map->phases[i]);
    }
    double ratio = entropy ? size / entropy : INFINITY;
    printf("size = %lld bytes, shannon = %d bytes, ratio = %0.3f\n", (long long) size, (int)entropy, ratio);
    
    return 0;
}
