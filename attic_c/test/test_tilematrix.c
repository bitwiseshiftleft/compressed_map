/** @file test_tilematrix.c
 * @brief Very simple test of tile matrix operations, to be
 * checked using SAGE.
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 */
#include "tile_matrix.h"
#include "bitset.h"
#include <stdio.h>
#include <string.h>

int main (int argc, char **argv) {
    const char *mode = "mul";
    if (argc > 1) mode = argv[1];

    if (!strcmp(mode,"test")) {
        int rows=35, match=47, cols=32, aug=16;

        tile_matrix_t ma[1], mb[1], mc[1];
        tile_matrix_init(ma,rows,match,aug);
        tile_matrix_init(mb,match,cols,aug);
        tile_matrix_init(mc,rows,cols,aug);
        tile_matrix_randomize(ma);
        tile_matrix_randomize(mb);
        tile_matrix_multiply_accumulate(mc,ma,mb);

        tile_matrix_print("ma",ma,1);
        tile_matrix_print("mb",mb,1);
        tile_matrix_print("mc",mc,1);

        tile_matrix_systematic_t sys;
        int err = tile_matrix_systematic_form(&sys,ma);
        tile_matrix_print("marr",ma,1);
        if (err) {
            printf("No systematic form\n");
        } else {
            tile_matrix_print("ma_sys", &sys.rhs,1);
        }
        tile_matrix_systematic_destroy(&sys);

    } else if (!strcmp(mode,"reduce")) {
        int rows=10000, cols=10000, ntrials=1;
        if (argc >= 3) rows = cols = atoll(argv[2]);
        if (argc >= 4) ntrials = atoll(argv[3]);
        tile_matrix_t ma[1];
        tile_matrix_init(ma,rows,cols,0);
        bitset_t ech = bitset_init(cols);

        int rank;
        for (; ntrials; ntrials--) {
            tile_matrix_randomize(ma);
            rank = tile_matrix_rref(ma,ech);
        }

        bitset_destroy(ech);
        printf("rk = %d\n", rank);
    } else if (!strcmp(mode,"sys")) {
        int rows=10000, cols=10000, ntrials=1;
        int inv=0;
        if (argc >= 3) rows = cols = atoll(argv[2]);
        if (argc >= 4) ntrials = atoll(argv[3]);
        tile_matrix_t ma[1];
        tile_matrix_init(ma,rows,cols,0);
        tile_matrix_systematic_t sys;

        for (int i=0; i<ntrials; i++) {
            tile_matrix_randomize(ma);
            int err = tile_matrix_systematic_form(&sys,ma);
            if (err==0) inv++;
            tile_matrix_systematic_destroy(&sys);
        }

        printf("full-rank = %d / %d\n", inv, ntrials);
    } else if (!strcmp(mode,"mul")) {
        int rows=10000, match=10000, cols=10000, ntrials=1;
        if (argc >= 3) rows = cols = match = atoll(argv[2]);
        if (argc >= 4) ntrials = atoll(argv[3]);
        tile_matrix_t ma[1], mb[1], mc[1];
        tile_matrix_init(ma,rows,match,0);
        tile_matrix_init(mb,match,cols,0);
        tile_matrix_init(mc,rows,cols,0);

        tile_matrix_randomize(ma);
        tile_matrix_randomize(mb);
        for (; ntrials; ntrials--) {
            tile_matrix_multiply_accumulate(mc,ma,mb);
        }
    } else if (!strcmp(mode,"rand")) {
        int rows=10000, cols=10000, ntrials=1;
        if (argc >= 3) rows = cols = atoll(argv[2]);
        if (argc >= 4) ntrials = atoll(argv[3]);
        tile_matrix_t ma[1], mb[1];
        tile_matrix_init(ma,rows,cols,0);
        tile_matrix_init(mb,rows,cols,0);

        for (; ntrials; ntrials--) {
            tile_matrix_randomize(ma);
            tile_matrix_randomize(mb);
        }
    } else {
        fprintf(stderr,"mode must be test, reduce, mul or rand\n");
    }

    return 0;
}
