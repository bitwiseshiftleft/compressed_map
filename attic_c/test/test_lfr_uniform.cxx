/** @file test_lfr_uniform.c
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 * @brief Test and bench uniform maps.
 */
#include <stdio.h>
#include "lfr_uniform.h"
#include <sys/time.h>
#include <string.h>
#include <sodium.h>
#include <math.h> // For INFINITY
#include <assert.h>
#include "util.h" // for le2ui

#ifndef LFR_BLOCKSIZE
#define LFR_BLOCKSIZE 4
#endif

static double now() {
    struct timeval tv;
    if (gettimeofday(&tv, NULL)) return 0;
    return tv.tv_sec + (double)tv.tv_usec / 1e6;
}

static void record(double *start, double *book) {
    double cur = now();
    if (cur > *start) {
        *book += cur - *start;
    }
    *start = cur;
}

void randomize(uint8_t *x, uint64_t seed, uint64_t nonce, size_t length) {
    uint8_t keybytes[crypto_stream_chacha20_KEYBYTES] = {0};
    uint8_t noncebytes[crypto_stream_chacha20_NONCEBYTES] = {0};
    memcpy(keybytes, &seed, sizeof(seed));
    memcpy(noncebytes, &nonce, sizeof(nonce));

    int ret = crypto_stream_chacha20(x, length, noncebytes, keybytes);
    if (ret) abort();
}

void usage(const char *fail, const char *me, int exitcode) {
    if (fail) fprintf(stderr, "Unknown argument: %s\n", fail);
    fprintf(stderr,"Usage: %s [--deficit 8] [--threads 0] [--augmented 8] [--blocks 2||--rows 32] [--blocks-max 0]\n", me);
    fprintf(stderr,"  [--blocks-step 10] [--exp 1.1] [--ntrials 100] [--verbose] [--seed 2] [--bail 3]\n");
    fprintf(stderr,"  [--tries 1] [--keylen 8] [--zeroize]\n");
    exit(exitcode);
}

int main(int argc, const char **argv) {
    long long blocks_min=2, blocks_max=-1, blocks_step=10, augmented=8, ntrials=100;
    uint64_t seed = 2;
    double ratio = 1.1;
    int is_exponential = 0, verbose=0, bail=3, nthreads=0, zeroize=0, tries=1;
    
    size_t keylen = 8;
        
    for (int i=1; i<argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg,"--augmented") && i<argc-1) {
            augmented = atoll(argv[++i]);
        } else if (!strcmp(arg,"--blocks") && i<argc-1) {
            blocks_min = atoll(argv[++i]);
	} else if (!strcmp(arg,"--bail") && i<argc-1) {
            bail = atoll(argv[++i]);
        } else if (!strcmp(arg,"--blocks-max") && i<argc-1) {
            blocks_max = atoll(argv[++i]);
        } else if (!strcmp(arg,"--rows") && i<argc-1) {
            blocks_min = _lfr_uniform_provision_columns(atoll(argv[++i])) / LFR_BLOCKSIZE / 8;
            if (blocks_min < 2) blocks_min = 2;
        } else if (!strcmp(arg,"--rows-max") && i<argc-1) {
            blocks_max = atoll(argv[++i]) / LFR_BLOCKSIZE / 8;
        } else if (!strcmp(arg,"--blocks-step") && i<argc-1) {
            blocks_step = atoll(argv[++i]);
            is_exponential = 0;
        } else if (!strcmp(arg,"--rows-step") && i<argc-1) {
            blocks_step = atoll(argv[++i]) / LFR_BLOCKSIZE / 8;
            is_exponential = 0;
        } else if (!strcmp(arg,"--keylen") && i<argc-1) {
            keylen = atoll(argv[++i]);
        } else if (!strcmp(arg,"--tries") && i<argc-1) {
            tries = atoll(argv[++i]);
        } else if (!strcmp(arg,"--zeroize")) {
            zeroize = 1;
        } else if (!strcmp(arg,"--exp")) {
            is_exponential = 1;
            if (i <argc-1) ratio = atof(argv[++i]);
        } else if (!strcmp(arg,"--ntrials") && i<argc-1) {
            ntrials = atoll(argv[++i]);
        } else if (!strcmp(arg,"--threads") && i<argc-1) {
            nthreads = atoll(argv[++i]);
        } else if (!strcmp(arg,"--seed") && i<argc-1) {
            seed = atoll(argv[++i]);
        } else if (!strcmp(arg,"--verbose")) {
            verbose = 1;
        } else {
            usage(argv[i], argv[0],1);
        }
    }
    (void)nthreads;
    
    if (blocks_max <= 0) blocks_max = blocks_min;
    
    if (augmented > 64) {
        printf("We don't support augmented > 64\n");
        return 1;
    }
    unsigned rows_max = _lfr_uniform_provision_max_rows(LFR_BLOCKSIZE*8*blocks_max);

    uint8_t  *keys   = (uint8_t*)malloc(rows_max*keylen);
    lfr_response_t *values = (lfr_response_t*)calloc(rows_max, sizeof(*values));
    if (keys == NULL || values == NULL) {
        printf("Can't allocate %lld key value pairs\n", (long long)rows_max);
        return 1;
    }
    
    if (blocks_min <= 1) {
        fprintf(stderr, "Must have at least 2 blocks\n");
        return 1;
    }
    
    if (blocks_min > blocks_max) {
        fprintf(stderr, "No blocks\n");
        return 1;
    }
    
    
    int successive_fails = 0;
    uint64_t mask = (augmented==64) ? -(uint64_t)1 : ((uint64_t)1 << augmented)-1;
    for (long long blocks=blocks_min; blocks <= blocks_max && (bail <= 0 || successive_fails < bail); ) {

        size_t rows = _lfr_uniform_provision_max_rows(LFR_BLOCKSIZE*8*blocks);
        double us_per_query = INFINITY, sps = INFINITY, us_per_build = INFINITY, ns_per_hash = INFINITY, ns_per_sample = INFINITY;

        size_t row_deficit = LFR_BLOCKSIZE*8*blocks - rows;
        lfr_salt_t salt;
        uint8_t salt_as_bytes[sizeof(salt)];
        randomize(salt_as_bytes, seed, blocks<<32 ^ 0xFFFFFFFF, sizeof(salt_as_bytes));
        salt = le2ui(salt_as_bytes, sizeof(salt_as_bytes));
        LibFrayed::Builder builder(rows,0,LFR_NO_COPY_DATA);
        builder.builder->max_tries = tries;
    
        double start, tot_construct=0, tot_query=0, tot_sample=0, tot_builder=0, ignored=0;
        size_t passes=0;
        bool did_ser_test=false;
        for (unsigned t=0; t<ntrials; t++) {
            start = now();

            builder.reset();
            randomize(keys, seed, blocks<<32 ^ t<<1,    rows*keylen);
            if (!zeroize) randomize((uint8_t*)values,seed,blocks<<32 ^ t<<1 ^ 1,rows*sizeof(*values));
            record(&start, &tot_sample);

            for (unsigned i=0; i<rows; i++) {
                builder.lookup(&keys[keylen*i], keylen) = values[i] & mask;
            }
            record(&start, &tot_builder);

            bool success = false;
            LibFrayed::UniformMap map;
            try {
                map = LibFrayed::UniformMap(builder, zeroize ? augmented : -1, nthreads);
                success = true;
            } catch (LibFrayed::BuildFailedException &e) {
                if (verbose) printf("Solve error\n");
            }
            record(&start, &tot_construct);

            if (success && !did_ser_test) {
                map = LibFrayed::UniformMap(map.serialize());
                did_ser_test = true;
            }
            record(&start,&ignored);
        
            int allpass = 1;
            for (unsigned i=0; i<rows && success; i++) {
                uint64_t ret = map.lookup(&keys[i*keylen], keylen);
                if (ret != (values[i] & mask)) {
                    if (verbose) printf("  Fail in row %lld: should be 0x%llx, actually 0x%llx\n",
                        (long long)i, (long long)(values[i] & mask), (long long)ret
                    );
                    allpass = 0;
                }
            }
            record(&start,&tot_query);
            if (allpass && success && verbose) printf("  Pass!\n");
            passes += success && allpass;
            record(&start, &tot_query);
        }

        if (passes) {
            us_per_query = tot_query * 1e6 / passes / rows;
            ns_per_sample = tot_sample * 1e9 / passes / rows;
            ns_per_hash = tot_builder * 1e9 / passes / rows;
            us_per_build = tot_construct * 1e6 / passes / rows;
	        successive_fails = 0;
        } else {
  	        successive_fails ++;
	    }
        if (tot_construct > 0) sps = passes / tot_construct;
        printf("Size %6d*%d*8 - %d x +%d pass rate = %4d / %4d = %5.1f%%, time/trial=%0.5f s, samp/row=%0.4f ns, ht/row=%0.4fns, build/row=%0.5f us, query/row=%0.5f us,  SPS=%0.3f\n",
            (int)blocks, (int)LFR_BLOCKSIZE, (int)row_deficit, (int)augmented, (int)passes,
            (int)ntrials, 100.0*passes/ntrials,
            (tot_construct+tot_builder+tot_sample)/ntrials, ns_per_sample, ns_per_hash, us_per_build, us_per_query,
            sps);
        fflush(stdout);
        
        if (is_exponential) {
            long long blocks2 = blocks * ratio;
            if (blocks2 == blocks) blocks2++;
            blocks = blocks2;
        } else {
            blocks += blocks_step;
        }
    }
    
    free(keys);
    free(values);
    
    return 0;
}
