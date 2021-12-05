/** @file tile_matrix.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Operations on matrices, represented by 8x8 submatrix "tiles".
 * @todo: document restrictions on all these functions, e.g. on matrix dimensions
 */
#ifndef __TILE_MATRIX_H__
#define __TILE_MATRIX_H__

#include <stdlib.h>
#include "tile_ops.h"
#include "bitset.h"

/** Matrix made of tiles */
typedef struct {
    size_t rows; /** Number of rows */
    size_t cols; /** Number of non-augmented columns */
    size_t aug_cols; /** Number of augmented columns */
    size_t stride; /** Row-to-row stride, measured in tiles */
    tile_t *data; /** Matrix data.  Row-major order, but the tiles themselves are column-major */
} tile_matrix_t;

/** Systematic form of a matrix, roughly of the form ( I | RHS ).
 * However, it's possible that some identity columns will be past
 * the beginning of the RHS.
 */
typedef struct {
    tile_matrix_t rhs; /** The non-identity component */
    bitset_t column_is_in_echelon; /** Bitset for which columns are in the identity-component */
} tile_matrix_systematic_t;

/**
 * Create a matrix with the given number of rows and columns.
 * The matrix is zeroized.
 * @return 0 on success, or -ENOMEM if the matrix is too large.
 */
int tile_matrix_init(tile_matrix_t *matrix, size_t rows, size_t cols, size_t augcols);

/** Destroy a matrix.  Free any intetrnal storage but not matrix itself
 * (which might be e.g. stack-allocated).
 */
void tile_matrix_destroy(tile_matrix_t *matrix);

/** Zeroize a matrix */
void tile_matrix_zeroize(tile_matrix_t *matrix);

/** Randomize the matrix using the C random() function */
void tile_matrix_randomize(tile_matrix_t *matrix);

/** Get a bit from the matrix's non-augmented section */
int tile_matrix_get_bit(const tile_matrix_t *matrix, size_t row, size_t cols);

/** Get a bit from the matrix's augmented section */
int tile_matrix_get_aug_bit(const tile_matrix_t *matrix, size_t row, size_t cols);

/** Print a matrix to stdout, with the given name, for debugging.
 * If for_sage, print in a way that can be pasted into sage
 */
void tile_matrix_print(const char *name, const tile_matrix_t *matrix, int for_sage);

/** Set out += a*b.  Out must not alias a or b. */
void tile_matrix_multiply_accumulate(
    tile_matrix_t *out,
    const tile_matrix_t *a,
    const tile_matrix_t *b
);

/** Set a row of the matrix.  Data and/or augdata can be NULL to indicate zero. */
void tile_matrix_set_row(tile_matrix_t *a, size_t row, const uint8_t *data, const uint8_t *augdata);

/** Copy rows a[rowa +: nrows] to b[rowb +: nrows] */
void tile_matrix_copy_rows(tile_matrix_t *a, const tile_matrix_t *b, size_t rowa, size_t rowb, size_t nrows);

/** Xor row a[rowa] to b[rowb] */
void tile_matrix_xor_row(tile_matrix_t *a, const tile_matrix_t *b, size_t rowa, size_t rowb);

/** Copy cols a[cola +: ncols] to b[colb +: ncols] */
void tile_matrix_copy_cols(tile_matrix_t *a, const tile_matrix_t *b, size_t cola, size_t colb, size_t ncols);

/** Zeroize rows a[rowa +: nrows] */
void tile_matrix_zeroize_rows(tile_matrix_t *a, size_t rowa, size_t nrows);

/** Zeroize cols a[cola +: ncols] */
void tile_matrix_zeroize_cols(tile_matrix_t *a, size_t cola, size_t ncols);

/** Xor in the augdata from b into a.  They must have the same sized augdata. */
void tile_matrix_xor_augdata(tile_matrix_t *a, const tile_matrix_t *b);

/** Resize the matrix by changing its number of rows (using realloc).
 * @return 0 on success, or -ENOMEM if out of memory.
 */
int tile_matrix_change_nrows(tile_matrix_t *matrix, size_t new_rows);

/** Move rows a[target +: nrows] to a[src +: nrows]
 * Zeroize the src rows, except the ones which have become target rows.
 */
void tile_matrix_move_rows(tile_matrix_t *a, size_t target, size_t src, size_t nrows);

/** For a very specific lfr_uniform use case.
 * Create a submatrix that just references a's memory, and contains rows
 * a[row_offset +: nrows]
 * The row_offset must be a multiple of TILE_SIZE.
 * Doesn't make sure that the extra rows (>nrows, if not aligned) are zero.
 * 
 * Don't destroy the submatrix; that would free the pointer to the submatrix.
 */
void _tile_aligned_submatrix(tile_matrix_t *sub, const tile_matrix_t *a, size_t nrows, size_t row_offset);

/**
 * Put the matrix in reduced row echelon form.
 * @return the rank of a
 */
size_t tile_matrix_rref(tile_matrix_t *a, bitset_t column_is_in_echelon);

/**
 * Echelonize the matrix a, then initialize sys to be its systematic form.
 * @return 0 on success; -1 or -ENOMEM on error.
 */
int tile_matrix_systematic_form(tile_matrix_systematic_t *sys, tile_matrix_t *a);

/** Destroy a systematic-form structure.  Frees the bitset and matrix but not sys itself. */
void tile_matrix_systematic_destroy(tile_matrix_systematic_t *sys);

/**
 * Create a trivial systematic matrix, with no columns in echelon.
 * @return 0 on success; -ENOMEM on error.
 */
int tile_matrix_trivial_systematic_form(tile_matrix_systematic_t *sys, size_t rows);

#endif // __TILE_MATRIX_H__
