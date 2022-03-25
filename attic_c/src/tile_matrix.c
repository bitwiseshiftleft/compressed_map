/** @file tile_matrix.c
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Tile matrix implementation.
 */
#include "tile_matrix.h"
#include "util.h"
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

/* Tile-based matrix calculation library, starting with inline tile manipulation
 * functions.
 *
 * The idea is to divide the matrix up into 8x8 tiles represented by 64-bit words.
 * Then we can improve the arithmetic density of the code by performing matrix
 * multiplication or reduction one tile at a time (or several by using the vector),
 * instead of one bit (or row of bits) at a time.
 * 
 * Could implement for different sizes depending on what vector unit is available,
 * but it seems like not a terrible strategy to just use 8x8, and to vectorize as
 * 8x8n.  In particular, vector permute / table instructions tend to operate on
 * bytes, which is convenient here.
 */

#include "tile_ops.h"

/** Return pseudoinverse of a tile. */
static inline tile_t tile_reduce (
    tile_t tile, /* The tile to reduce */
    tile_t *permp, // map to adjust tiles down the column before multiplying to eliminate them
    tile_edge_t rows_avail, /* Mask of which rows which are eligible to reduce */
    tile_edge_t *columns_echelon /** Output mask of which columns are now in echelon. */
) {
    tile_t aug = tile_identity();
    tile_t perm = tile_zero();
    tile_edge_t ech = tile_edge_zero();
    
    for (unsigned col=0; col<TILE_SIZE; col++) {
        /* Eliminate column by column. */
        tile_t rowmask = tile_broadcast_edge_as_col(rows_avail);
        if (tile_is_zero(tile & rowmask)) break;
        tile_t colmask = tile_col_mask(col);
        tile_t coldata = tile & rowmask & colmask;
        if (tile_is_zero(coldata)) continue;

        ech |= (tile_edge_t)1<<col;
        unsigned row_found = bit_index_to_row(tile_get_first_nonzero_entry(coldata));
        unsigned row = __builtin_ctzll(rows_avail);
        if (row_found != row) {
            tile = tile_swap_rows(tile,row,row_found);
            aug = tile_swap_rows(aug,row,row_found);
        }

        rows_avail ^= 1<<row; // &~, but it's currently set so ^= is the same

        perm ^= tile_single_bit(col,row,1);
        tile_t rows_affected = tile_broadcast_col(tile^tile_single_bit(row,col,1),col);
        tile ^= tile_broadcast_row(tile,row) & rows_affected; // clear the tile
        aug ^= tile_broadcast_row(aug,row) & rows_affected; // clear the augmented tile
    }
    *permp = perm;
    *columns_echelon = ech;
    return aug;
}

/** Erase the whole matrix */
void tile_matrix_zeroize(tile_matrix_t *matrix) {
    memset(matrix->data,0,
        TILES_SPANNING(matrix->rows)
        * matrix->stride
        * sizeof(tile_t)
    );
}

int tile_matrix_init(tile_matrix_t *matrix, size_t rows, size_t cols, size_t aug_cols) {
    size_t trows = TILES_SPANNING(rows), tstride = TILES_SPANNING(cols) + TILES_SPANNING(aug_cols);
    matrix->data = calloc(sizeof(tile_t),trows*tstride);
    if (matrix->data != NULL || rows == 0 || tstride == 0) {
        matrix->rows = rows;
        matrix->cols = cols;
        matrix->aug_cols = aug_cols;
        matrix->stride = tstride;
        return 0;
    } else {
        memset(matrix,0,sizeof(*matrix));
        return ENOMEM;
    }
}

int tile_matrix_change_nrows(tile_matrix_t *matrix, size_t new_rows) {
    if (new_rows == matrix->rows) {
        return 0;
    } else if (new_rows > matrix->rows) {
        size_t trows = TILES_SPANNING(new_rows);
        matrix->data = realloc(matrix->data,sizeof(tile_t)*trows*matrix->stride);
        if (matrix->data == NULL) {
            tile_matrix_destroy(matrix);
            return ENOMEM;
        }
        size_t new_rows_up = round_up(new_rows, TILE_SIZE);
        tile_matrix_zeroize_rows(matrix, matrix->rows, new_rows_up-matrix->rows);
    } else {
        tile_matrix_zeroize_rows(matrix, new_rows, matrix->rows-new_rows);
    }
    matrix->rows = new_rows;
    return 0;
}

void _tile_aligned_submatrix(tile_matrix_t *sub, const tile_matrix_t *a, size_t nrows, size_t row_offset) {
    assert(row_offset % TILE_SIZE == 0);
    size_t stride = sub->stride = a->stride;
    sub->cols = a->cols;
    sub->aug_cols = a->aug_cols;
    sub->data = a->data + stride*(row_offset/TILE_SIZE);
    sub->rows = nrows;
}

void tile_matrix_move_rows(tile_matrix_t *a, size_t target, size_t src, size_t nrows) {
    if (target == src) return;
    tile_matrix_copy_rows(a,a,target,src,nrows);

    // OK, now clear the source
    if (src >= target+nrows || target >= src+nrows) {
        // they're disjoint, so clear cleanly
        tile_matrix_zeroize_rows(a, src, nrows);
    } else if (target < src) {
        // moved down
        tile_matrix_zeroize_rows(a, target+nrows, src-target);
    } else {
        // moved up
        tile_matrix_zeroize_rows(a, src, target-src);
    }
}

void tile_matrix_randomize(tile_matrix_t *matrix) {
    size_t rows = matrix->rows, cols = matrix->cols, aug_cols = matrix->aug_cols;
    size_t trows = TILES_SPANNING(rows), tcols = TILES_SPANNING(cols);
    size_t taug = TILES_SPANNING(aug_cols), tstride = matrix->stride, aug_off = tcols;
    tile_t last_row_mask = (rows % TILE_SIZE) ? tile_mask_of_rows_less_than(rows % TILE_SIZE) : tile_full();
    tile_t last_col_mask = (cols % TILE_SIZE) ? tile_mask_of_cols_less_than(cols % TILE_SIZE) : tile_full();
    tile_t last_aug_mask = (aug_cols % TILE_SIZE) ? tile_mask_of_cols_less_than(aug_cols % TILE_SIZE) : tile_full();
    for (size_t trow=0; trow<trows; trow++) {
        for (size_t tcol=0; tcol<tcols; tcol++) {
            tile_t t = tile_random();
            if (trow == trows-1) t &= last_row_mask;
            if (tcol == tcols-1) t &= last_col_mask;
            matrix->data[tcol + trow*tstride] = t;
        }
        for (size_t tcol=0; tcol<taug; tcol++) {
            tile_t t = tile_random();
            if (trow == trows-1) t &= last_row_mask;
            if (tcol == taug-1) t &= last_aug_mask;
            matrix->data[tcol + aug_off + trow*tstride] = t;
        }
    }
}

int tile_matrix_get_bit(const tile_matrix_t *a, size_t row, size_t col) {
    size_t tstride = a->stride;
    tile_t aij = a->data[(row/TILE_SIZE)*tstride + col/TILE_SIZE];
    return tile_get_bit(aij, row%TILE_SIZE, col%TILE_SIZE);
}

int tile_matrix_get_aug_bit(const tile_matrix_t *a, size_t row, size_t col) {
    size_t tstride = a->stride;
    tile_t aij = a->data[(row/TILE_SIZE)*tstride + TILES_SPANNING(a->cols) + col/TILE_SIZE];
    return tile_get_bit(aij, row%TILE_SIZE, col%TILE_SIZE);
}

void tile_matrix_destroy(tile_matrix_t *matrix) {
    free(matrix->data);
    memset(matrix,0,sizeof(*matrix));
}

/* Perform the matrix row operation: c[0:ntiles-1] += a*b[0:ntiles-1] */
static __attribute__((noinline)) void tile_matrix_rowop(tile_t *c, tile_t a, const tile_t *b, size_t ntiles) {
    if (tile_is_zero(a)) return;
    tile_precomputed_t apre = tile_precompute_vmul(a);
    const size_t TVL = TILE_VECTOR_LENGTH;

    for (; ntiles >= TVL; ntiles -= TVL, c += TVL, b += TVL) {
        tile_write_v(c, tile_read_v(c) ^ tile_vmul(&apre, tile_read_v(b)));
    }
    if (ntiles > 0) {
        tile_write_vpartial(c, ntiles,
            tile_read_vpartial(c, ntiles)
            ^ tile_vmul(&apre, tile_read_vpartial(b, ntiles)));
    }
}

/* Perform the matrix row operation: c[0:ntiles] = a*c[0:ntiles] */
static __attribute__((noinline)) void tile_matrix_rowtimes(tile_t *c, tile_t a, size_t ntiles) {
    tile_precomputed_t apre = tile_precompute_vmul(a);

    const size_t TVL = TILE_VECTOR_LENGTH;
    for (; ntiles >= TVL; ntiles -= TVL, c += TVL) {
        tile_write_v(c, tile_vmul(&apre, tile_read_v(c)));
    }
    if (ntiles > 0) {
        tile_write_vpartial(c, ntiles, tile_vmul(&apre, tile_read_vpartial(c, ntiles)));
    }
}

void tile_matrix_multiply_accumulate(tile_matrix_t *out, const tile_matrix_t *a, const tile_matrix_t *b) {
    assert(a->cols == b->rows);
    assert(b->cols == out->cols);
    assert(a->rows == out->rows);
    assert(b->aug_cols <= out->aug_cols);
    assert(a->aug_cols <= out->aug_cols);

    size_t rows=a->rows, cols=b->cols, match = a->cols;
    size_t trows = TILES_SPANNING(rows), tstride_b = b->stride, tmatch = TILES_SPANNING(match);
    size_t tstride_a = a->stride, tstride_c = out->stride;
    size_t oplen = TILES_SPANNING(cols) + TILES_SPANNING(b->aug_cols);
    size_t augoff_out = TILES_SPANNING(out->cols), augoff_a = TILES_SPANNING(a->cols);
    size_t t_auglen = TILES_SPANNING(a->aug_cols);

    for (size_t i=0; i<trows; i++) {
        for (size_t j=0; j<tmatch; j++) {
            tile_t aij = a->data[i*tstride_a+j];
            tile_matrix_rowop(&out->data[i*tstride_c], aij, &b->data[j*tstride_b], oplen);
        }
        for (size_t j=0; j<t_auglen; j++) {
            out->data[i*tstride_c+augoff_out+j] ^= a->data[i*tstride_a+augoff_a+j];
        }
    }
}

/** Swap a[r1:r1+nrows-1] with a[r2:r2+nrows-1]
 * If they aren't disjoint, then move the later rows together as a block; the earlier rows
 * will end up in some order at the end.
 */
static void tile_matrix_swap_rows(tile_matrix_t *a, size_t r1, size_t r2, size_t nrows, size_t start_col) {
    // Make r1 earlier
    if (r1 == r2) return;
    if (r1 > r2) { size_t tmp = r1; r1=r2; r2=tmp; }
    assert(r1 < r2);

    size_t stride = a->stride;
    while (nrows > 0) {
        size_t tr1 = r1/TILE_SIZE, tr2=r2/TILE_SIZE;
        size_t lr1 = r1%TILE_SIZE, lr2=r2%TILE_SIZE;
        size_t cando = TILE_SIZE - ((lr1<lr2) ? lr2 : lr1);
        if (cando > nrows) cando = nrows;
        if (cando > r2-r1) cando = r2-r1;

        if (tr1 == tr2) {
            /* Implement swap within a row as rowop */
            tile_t perm = tile_bulk_swap_rows(tile_identity(), lr1, lr2, cando);
            tile_matrix_rowtimes(&a->data[tr1*stride+start_col], perm, stride-start_col);
        } else {
            for (size_t i=start_col; i<stride; i++) {
                tile2_bulk_swap_rows(&a->data[tr1*stride+i], &a->data[tr2*stride+i], lr1, lr2, cando);
            }
        }
        nrows -= cando;
        r1 += cando;
        r2 += cando;
    }
}

static void tile_matrix_copy_column(tile_matrix_t *b, const tile_matrix_t *a, size_t colb, size_t cola) {
    /* Copy a[cola] to b[colb] */
    size_t astride = a->stride, bstride = b->stride;
    size_t trows = TILES_SPANNING(a->rows), tcola=cola/TILE_SIZE, tcolb=colb/TILE_SIZE;
    assert(trows == TILES_SPANNING(b->rows));
    for (size_t trow=0; trow<trows; trow++) {
        b->data[bstride*trow + tcolb] = tile_copy_col(
            b->data[bstride*trow + tcolb],
            a->data[astride*trow + tcola],
            colb % TILE_SIZE, cola % TILE_SIZE
        );
    }
}

static void tile_matrix_copy_one_rowgroup (
    tile_matrix_t *b, const tile_matrix_t *a, size_t rowb, size_t rowa, size_t nrows
) {
    /* As tile_matrix_copy_rows, but for only one group of rows at a time */
    size_t astride = a->stride, bstride = b->stride;
    size_t tcols =  TILES_SPANNING(a->cols) + TILES_SPANNING(a->aug_cols);
    assert(tcols == TILES_SPANNING(b->cols) + TILES_SPANNING(b->aug_cols));
    size_t trowb = rowb/TILE_SIZE, trowa = rowa/TILE_SIZE;
    for (size_t tcol=0; tcol<tcols; tcol++) {
        b->data[bstride*trowb + tcol] = tile_bulk_copy_rows (
            b->data[bstride*trowb + tcol],
            a->data[astride*trowa + tcol],
            rowb % TILE_SIZE, rowa % TILE_SIZE, nrows
        );
    }
}

static void tile_matrix_zeroize_one_rowgroup(tile_matrix_t *b, size_t rowb, size_t nrows) {
    size_t bstride = b->stride;
    size_t tcols =  TILES_SPANNING(b->cols) + TILES_SPANNING(b->aug_cols);
    size_t trowb = rowb/TILE_SIZE;
    if (nrows == TILE_SIZE) {
        memset((uint8_t*)&b->data[bstride*trowb], 0, bstride*sizeof(tile_t));
    } else {
        for (size_t tcol=0; tcol<tcols; tcol++) {
            b->data[bstride*trowb + tcol] &=~ tile_row_bulk_mask(rowb % TILE_SIZE,nrows);
        }
    }
}

static void tile_matrix_zeroize_one_colgroup(tile_matrix_t *b, size_t colb, size_t ncols) {
    size_t bstride = b->stride;
    size_t trows = TILES_SPANNING(b->rows);
    size_t tcolb = colb/TILE_SIZE;
    for (size_t trow=0; trow<trows; trow++) {
        b->data[bstride*trow + tcolb] &=~ tile_col_bulk_mask(colb,ncols);
    }
}

void tile_matrix_zeroize_rows(tile_matrix_t *b, size_t rowb, size_t nrows) {
    /* Zeroize b[rowb +: nrows] */
    while (nrows > 0) {
        size_t lrb = rowb%TILE_SIZE;
        size_t cando = TILE_SIZE - lrb;
        if (cando > nrows) cando = nrows;
        tile_matrix_zeroize_one_rowgroup(b,rowb,cando);
        rowb += cando;
        nrows -= cando;
    }
}

void tile_matrix_zeroize_cols(tile_matrix_t *b, size_t colb, size_t ncols) {
    /* Zeroize b[colb +: ncols] */
    while (ncols > 0) {
        size_t lrb = colb%TILE_SIZE;
        size_t cando = TILE_SIZE - lrb;
        if (cando > ncols) cando = ncols;
        tile_matrix_zeroize_one_colgroup(b,lrb,cando);
        colb += cando;
        ncols -= cando;
    }
}

void tile_matrix_copy_rows(tile_matrix_t *b, const tile_matrix_t *a, size_t rowb, size_t rowa, size_t nrows) {
    /* Copy a[rowa +: nrows] to b[rowb +: nrows] */
    if (a==b && rowa == rowb) return;
    if (a != b || rowb <= rowa) {
        while (nrows > 0) {
            size_t lra = rowa%TILE_SIZE, lrb = rowb%TILE_SIZE;
            size_t cando = TILE_SIZE - ((lra<lrb) ? lrb : lra);
            if (cando > nrows) cando = nrows;
            tile_matrix_copy_one_rowgroup(b,a,rowb,rowa,cando);
            rowa += cando;
            rowb += cando;
            nrows -= cando;
        }
    } else {
        // copying a block forward in the matrix -- do it from the end
        rowa += nrows-1;
        rowb += nrows-1;
        while (nrows > 0) {
            size_t lra = rowa%TILE_SIZE,   lrb = rowb%TILE_SIZE;
            size_t cando = (lrb > lra) ? (lra+1) : (lrb+1);
            if (cando > nrows) cando = nrows;
            tile_matrix_copy_one_rowgroup(b,a,rowb-cando+1,rowa-cando+1,cando);
            rowa -= cando;
            rowb -= cando;
            nrows -= cando;
        }
    }
}

void tile_matrix_xor_row(tile_matrix_t *b, const tile_matrix_t *a, size_t rowb, size_t rowa) {
    /* Xor a[rowa] into b[rowb] */
    size_t astride = a->stride, bstride = b->stride;
    size_t tcols =  TILES_SPANNING(b->cols) + TILES_SPANNING(b->aug_cols);
    size_t trowa = rowa/TILE_SIZE, trowb = rowb/TILE_SIZE;
    for (size_t tcol=0; tcol<tcols; tcol++) {
        b->data[bstride*trowb + tcol] =
            tile_bulk_xor_rows(b->data[bstride*trowb + tcol],
                a->data[astride*trowa + tcol],
                rowb % TILE_SIZE, rowa % TILE_SIZE, 1);
    }
}


static void tile_matrix_copy_one_colgroup (
    tile_matrix_t *b, const tile_matrix_t *a, size_t colb, size_t cola, size_t ncols
) {
    // As tile_matrix_copy_rows, but for only one group of rows at a time 
    size_t astride = a->stride, bstride = b->stride;
    size_t trows = TILES_SPANNING(a->rows);
    size_t tcolb = colb/TILE_SIZE, tcola = cola/TILE_SIZE;
    for (size_t trow=0; trow<trows; trow++) {
        b->data[bstride*trow + tcolb] = tile_bulk_copy_cols(
            b->data[bstride*trow + tcolb],
            a->data[astride*trow + tcola],
            colb % TILE_SIZE, cola % TILE_SIZE, ncols
        );
    }
}

void tile_matrix_copy_cols(tile_matrix_t *b, const tile_matrix_t *a, size_t colb, size_t cola, size_t ncols) {
    /* Copy a[rowa +: nrows] to b[rowb +: nrows] */
    if (a==b && cola == colb) return;

/*
    // Little-endian code, but it isn't much faster
    size_t stridea = a->stride*sizeof(tile_t), strideb = b->stride*sizeof(tile_t);
    uint8_t *datab = (uint8_t*)b->data;
    const uint8_t *dataa = (uint8_t*)a->data;

    datab += (colb/TILE_SIZE)*sizeof(tile_t) + (colb % TILE_SIZE)*(TILE_SIZE/8);
    dataa += (cola/TILE_SIZE)*sizeof(tile_t) + (cola % TILE_SIZE)*(TILE_SIZE/8);

    if (a==b) { // use memmove
        for (size_t row=0; row<TILES_SPANNING(a->rows); row++) {
            memmove(datab + strideb*row, dataa + stridea*row, ncols * (TILE_SIZE/8));
        }
    } else {
        for (size_t row=0; row<TILES_SPANNING(a->rows); row++) {
            memcpy(datab + strideb*row, dataa + stridea*row, ncols * (TILE_SIZE/8));
        }
    }
*/

    if (a != b || colb <= cola) {
        while (ncols > 0) {
            size_t lra = cola%TILE_SIZE,   lrb = colb%TILE_SIZE;
            size_t cando = TILE_SIZE - ((lra<lrb) ? lrb : lra);
            if (cando > ncols) cando = ncols;
            tile_matrix_copy_one_colgroup(b,a,colb,cola,cando);
            cola += cando;
            colb += cando;
            ncols -= cando;
        }
    } else {
        // copying a block forward in the matrix -- do it from the end
        cola += ncols-1;
        colb += ncols-1;
        while (ncols > 0) {
            size_t lra = cola%TILE_SIZE,   lrb = colb%TILE_SIZE;
            size_t cando = (lrb > lra) ? (lra+1) : (lrb+1);
            if (cando > ncols) cando = ncols;
            tile_matrix_copy_one_colgroup(b,a,colb-cando+1,cola-cando+1,cando);
            cola -= cando;
            colb -= cando;
            ncols -= cando;
        }
    }
}

void tile_matrix_xor_augdata(tile_matrix_t *a, const tile_matrix_t *b) {
    size_t tcolsa = TILES_SPANNING(a->cols), tcolsb = TILES_SPANNING(b->cols);
    size_t trows = TILES_SPANNING(b->rows), taugs = TILES_SPANNING(a->aug_cols);
    size_t astride = a->stride, bstride = b->stride;
    assert(a->rows >= b->rows);
    assert(a->aug_cols == b->aug_cols);

    for (size_t trow=0; trow<trows; trow++) {
        for (size_t taug=0; taug<taugs; taug++) {
            a->data[astride*trow+tcolsa+taug] ^= b->data[bstride*trow+tcolsb+taug];
        }
    }
}

void tile_matrix_set_row(tile_matrix_t *a, size_t row, const uint8_t *data, const uint8_t *augdata) {
    assert(TILE_SIZE%8 == 0);
    const size_t TILE_BYTES = TILE_SIZE/8;
    assert(TILE_BYTES == sizeof(tile_edge_t));
    assert(TILE_BYTES == 1); // if not, need to make sure we don't index off the end of augdata

    tile_t *r = &a->data[a->stride*(row/TILE_SIZE)];
    size_t tcols = TILES_SPANNING(a->cols), taug = TILES_SPANNING(a->aug_cols);
    int subrow = row % TILE_SIZE;

    // set the main part
    tile_edge_t last_col_mask = (a->cols % TILE_SIZE) ? ((tile_edge_t)1<<(a->cols%TILE_SIZE)) - 1 : tile_edge_full();
    for (size_t i=0; i<tcols; i++) {
        tile_edge_t thisdata = (data==NULL) ? tile_edge_zero() : le2ui(&data[i*TILE_BYTES], TILE_BYTES);
        if (i==tcols-1) thisdata &= last_col_mask;
        r[i] = tile_set_row(r[i], subrow, thisdata);
    }

    // set the aug
    tile_edge_t last_aug_mask = (a->aug_cols % TILE_SIZE) ? ((tile_edge_t)1<<(a->aug_cols%TILE_SIZE)) - 1 : tile_edge_full();
    for (size_t i=0; i<taug; i++) {
        tile_edge_t thisdata = (augdata==NULL) ? tile_edge_zero() : le2ui(&augdata[i*TILE_BYTES], TILE_BYTES);
        if (i==tcols-1) thisdata &= last_aug_mask;
        r[i+tcols] = tile_set_row(r[i+tcols], subrow, thisdata);
    }
}

/*****************************************************
 * High-level matrix operations
 *****************************************************/

void tile_matrix_print(const char *name, const tile_matrix_t *matrix, int for_sage) {
    if (for_sage) {
        printf("%s = matrix(GF(2),[", name);
        for (unsigned row=0; row<matrix->rows; row++) {
            printf("%s\n  [", row ? ",": "");
            for (unsigned col=0; col<matrix->cols; col++) {
                printf("%s%d", col ? ", " : "", tile_matrix_get_bit(matrix,row,col));
            }
            printf("%s   ", matrix->cols ? "," : " ");
            for (unsigned col=0; col<matrix->aug_cols; col++) {
                printf("%s%d", col ? ", " : "", tile_matrix_get_aug_bit(matrix,row,col));
            }
            printf("]");
        }
        printf("\n])\n\n");
    } else {
        printf("# tile matrix %s:\n", name);
        for (unsigned row=0; row<matrix->rows; row++) {
            printf("  # ");
            for (unsigned col=0; col<matrix->cols; col++) {
                printf("%s", tile_matrix_get_bit(matrix,row,col) ? "1" : " ");
            }
            printf(" # ");
            for (unsigned col=0; col<matrix->aug_cols; col++) {
                printf("%s", tile_matrix_get_aug_bit(matrix,row,col) ? "1" : " ");
            }
            printf("\n");
        }
        printf("\n");
    }
}

size_t tile_matrix_rref(tile_matrix_t *a, bitset_t column_is_in_echelon) {
    size_t rows = a->rows, cols = a->cols;
    size_t trows = TILES_SPANNING(rows), tcols = TILES_SPANNING(cols), tstride = a->stride;
    size_t ttotal = tcols + TILES_SPANNING(a->aug_cols);
    bitset_clear_all(column_is_in_echelon, cols);

    if (trows == 0) return 0; // trivial

    size_t rank = 0;

    /* Put the tile-columns into echelon form, one after another */
    for (size_t tcol=0; tcol<tcols; tcol++) {
        /* Which columns have we echelonized in this loop? */
        tile_edge_t ech = tile_edge_zero();
        tile_t perm_matrix_cumulative = tile_zero();

        /* We want to echelonize TILE_SIZE rows if possible.  To do this, we select as
         * the active tile-row one which has TILE_SIZE rows not in the current echelon
         * structure, if possible.  Since the rows in echelon get moved to the beginning,
         * this is either trow_min or trow_min + 1.
         */
        ssize_t trow_min = rank/TILE_SIZE, trow_begin = trow_min + (rank % TILE_SIZE != 0);
        if (trow_min   >= (ssize_t)trows) break; // Done!
        if (trow_begin >= (ssize_t)trows) trow_begin=trow_min;
        ssize_t trow=trow_begin;

        /* For brevity, make pointers to the main tile we're working with */
        tile_t *active = &a->data[trow_begin*tstride+tcol];
        size_t active_length = ttotal-tcol;

        /* Massage the active row so that it can eliminate every column */
        do {
            /* Which single rows are available?
             * All of them, unless we're on the first available tile-row
             */
            int first_available_row=0;
            if (trow*TILE_SIZE < (ssize_t)rank) {
                first_available_row = rank - trow*TILE_SIZE;
            }
            tile_edge_t rows_avail = tile_edge_full() & -(1<<first_available_row);

            /* Make pointers to the current tile we're looking at to get more columns */
            tile_t *working = &a->data[trow*tstride+tcol];

            /* If it's not the first, apply our progress so far */
            if (trow != trow_begin) {
                tile_t factor = tile_mul(*working, perm_matrix_cumulative);
                tile_matrix_rowop(working, factor, active, active_length);
            }

            /* Row-reduce the current tile */
            tile_edge_t ech_new = 0;
            tile_t perm, aug = tile_reduce(*working, &perm, rows_avail, &ech_new);

            if (ech_new) {
                /* We got some new columns.  Apply the operation to the rest of the row */
                tile_matrix_rowtimes(working, aug, active_length);

                if (trow != trow_begin) {
                    /* Eliminate these columns from the active row */
                    tile_t factor = tile_mul(*active, perm);
                    tile_matrix_rowop(active, factor, working, active_length);

                    /* Swap them into the active row */
                    int nech = __builtin_popcountll(ech), nech_new = __builtin_popcountll(ech_new);
                    perm_matrix_cumulative = tile_bulk_copy_cols(perm_matrix_cumulative, perm, nech, first_available_row, nech_new);
                    // tile2_bulk_swap_cols(&perm_matrix_cumulative, &perm, nech, first_available_row, nech_new);
                    for (size_t c=0; c<active_length; c++) {
                        tile2_bulk_swap_rows(&active[c], &working[c], nech, first_available_row, nech_new);
                    }
                    ech |= ech_new;
                } else {
                    perm_matrix_cumulative = perm;
                    ech = ech_new;
                }
            }

            /* next row */
            trow++;
            if (trow >= (ssize_t)trows) trow=trow_min; /* wrap around */
        } while (trow != trow_begin && ech != tile_edge_full());

        /* OK, we now have a tile which echelonizes all the selected columns.  Eliminate them. */
        for (ssize_t trow=0; trow<(ssize_t)trows; trow++) {
            if (trow==trow_begin) continue;
            tile_t factor = tile_mul(a->data[trow*tstride+tcol], perm_matrix_cumulative);
            tile_matrix_rowop(&a->data[trow*tstride+tcol], factor, active, active_length);
        }

        ssize_t begin = (trow_begin*TILE_SIZE < (ssize_t)rank) ? (ssize_t)rank : trow_begin*TILE_SIZE;
        if (!tile_is_zero(*active &~ tile_identity())) {
            /* The working tile is permuted, because it picked up some rows from one tile-row,
             * and other rows from another tile-row.  Unepermute it to make sure it's in REF */
            tile_edge_t rows_avail = tile_edge_full() & -(1<<(begin%TILE_SIZE));
            tile_edge_t ech_new;
            tile_t perm, pseudoinv = tile_reduce(*active, &perm, rows_avail, &ech_new);
            tile_matrix_rowtimes(active, pseudoinv, active_length);
        }

        /* Swap the active row into place at the beginning of the matrix */
        int nech = __builtin_popcountll(ech);
        tile_matrix_swap_rows(a, rank, begin, nech, tcol);
        rank += nech;

        /* Mark the echelonized columns */
        if (column_is_in_echelon != NULL) {
            for (unsigned i=0; i<TILE_SIZE/8; i++) {
                bitset_toggle_byte(column_is_in_echelon, tcol*(TILE_SIZE/8) + i, ech>>(8*i));
            }
        }
    }

    return rank;
}

int tile_matrix_systematic_form(tile_matrix_systematic_t *sys, tile_matrix_t *a) {
    /* Allocate the column bitset */
    sys->column_is_in_echelon = bitset_init(a->cols);
    if (!sys->column_is_in_echelon) {
        memset(sys,0,sizeof(*sys));
        return ENOMEM;
    }

    size_t rank = tile_matrix_rref(a, sys->column_is_in_echelon);
    if (rank < a->rows) {
        /* Not enough rank */
        bitset_destroy(sys->column_is_in_echelon);
        memset(sys,0,sizeof(*sys));
        return -1;
    }

    if (tile_matrix_init(&sys->rhs, a->rows, a->cols-rank, a->aug_cols)) {
        /* No memeory */
        bitset_destroy(sys->column_is_in_echelon);
        memset(sys,0,sizeof(*sys));
        return ENOMEM;
    }

    /* Copy the non-echelon part of the columns */
    for (size_t csrc=0,cdst=0; csrc<a->cols; csrc++) {
        if (!bitset_test_bit(sys->column_is_in_echelon, csrc)) {
            assert(cdst <= sys->rhs.cols);
            tile_matrix_copy_column(&sys->rhs, a, cdst, csrc);
            cdst++;
        }
    }

    /* Copy the augmented component */
    size_t a_aug = TILES_SPANNING(a->cols), a_stride=a->stride;
    size_t o_aug = TILES_SPANNING(sys->rhs.cols), o_stride=sys->rhs.stride;
    for (size_t trow=0; trow<TILES_SPANNING(a->rows); trow++) {
        for (size_t taug=0; taug<TILES_SPANNING(a->aug_cols); taug++) {
            sys->rhs.data[trow*o_stride+o_aug+taug] = a->data[trow*a_stride+a_aug+taug];
        }
    }

    return 0;
}

void tile_matrix_systematic_destroy(tile_matrix_systematic_t *sys) {
    bitset_destroy(sys->column_is_in_echelon);
    sys->column_is_in_echelon = NULL;
    tile_matrix_destroy(&sys->rhs);
}

int tile_matrix_trivial_systematic_form(tile_matrix_systematic_t *sys, size_t rows) {
    memset(sys,0,sizeof(*sys));
    sys->column_is_in_echelon = bitset_init(rows);
    if (sys->column_is_in_echelon == NULL) {
        return ENOMEM;
    }
    sys->rhs.cols = rows;
    return 0;
}
