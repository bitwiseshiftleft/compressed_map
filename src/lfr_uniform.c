#include "util.h"
#include "lfr_uniform.h"
#include "tile_matrix.h"
#include <string.h>
#include <errno.h>

#if LFR_THREADED
#include <pthread.h>
#include <sys/sysctl.h>
#endif

#if LFR_BLOCKSIZE==1
typedef uint8_t lfr_uniform_block_t;
#elif LFR_BLOCKSIZE==2
typedef uint16_t lfr_uniform_block_t;
#elif LFR_BLOCKSIZE==4
typedef uint32_t lfr_uniform_block_t;
#elif LFR_BLOCKSIZE==8
typedef uint64_t lfr_uniform_block_t;
#else
#error "Need LFR_BLOCKSIZE in [1,2,4,8]"
#endif

extern const unsigned int LFR_META_SAMPLE_BYTES;


/*************************************************
 * Start of code specific to frayed ribbon shape *
 *************************************************/

/** Block indices */
typedef uint32_t lfr_uniform_block_index_t;

#ifndef LFR_OVERPROVISION
#define LFR_OVERPROVISION 1024
#endif

static const size_t EXTRA_ROWS = 8;
size_t API_VIS _lfr_uniform_provision_columns(size_t rows) {
    size_t cols = rows + EXTRA_ROWS;
#if LFR_OVERPROVISION
    cols += cols/LFR_OVERPROVISION;
#endif
    cols += (-cols) % (8*LFR_BLOCKSIZE);
    if (cols <= 8*LFR_BLOCKSIZE) cols = 16*LFR_BLOCKSIZE;
    return cols;
}

size_t API_VIS _lfr_uniform_provision_max_rows(size_t cols) {
    size_t cols0 = cols;
    (void)cols0;
#if LFR_OVERPROVISION
    cols -= cols / (LFR_OVERPROVISION+1);
#endif
    if (cols <= EXTRA_ROWS) return 0;
    if (_lfr_uniform_provision_columns(cols-EXTRA_ROWS) > cols0) cols--;
    assert(_lfr_uniform_provision_columns(cols-EXTRA_ROWS) <= cols0);
    return cols - EXTRA_ROWS;
}

/** The main sampling function: given a seed, sample block positions */
static inline void _lfr_uniform_sample_block_positions (
    lfr_uniform_block_index_t out[2],
    size_t nblocks,
    uint32_t stride_seed32,
    uint32_t a_seed32
) {
    /* Parse the seed into uints */
    uint64_t stride_seed = stride_seed32;
    uint64_t a_seed      = a_seed32;
    __uint128_t nblocks_huge = nblocks;

    // calculate log(nblocks)<<48 in a smooth way
    uint64_t k = high_bit(nblocks);
    uint64_t smoothlog = (k<<48) + (nblocks<<(48-k)) - (1ull<<48);
    uint64_t leading_coefficient = (12ull<<8)/LFR_BLOCKSIZE; // experimentally determined
    uint64_t num = smoothlog * leading_coefficient; 

#if LFR_OVERPROVISION
    stride_seed |= (1ull<<33) / LFR_OVERPROVISION; // | instead of + because it can't overflow
#endif
    uint64_t den = ((stride_seed * stride_seed) * nblocks_huge) >> 32;
#if (!LFR_OVERPROVISION) || (LFR_OVERPROVISION > 1ull<<16)
    den++; // to prevent it from being 0
#endif
    uint64_t b_seed = (num / den + a_seed) & 0xFFFFFFFF; // den can't be 0 because stride_seed is adjusted

    uint64_t a = (a_seed * nblocks_huge)>>32, b = (b_seed * nblocks_huge)>>32;
    if (a==b && ++b >= nblocks) b=0;
    out[0] = a;
    out[1] = b;
}

/***********************************************
 * End of code specific to frayed ribbon shape *
 ***********************************************/

void API_VIS lfr_builder_destroy(lfr_builder_t matrix) {
    free(matrix->relations);
    memset(matrix,0,sizeof(*matrix));
}

void API_VIS lfr_builder_reset(lfr_builder_t matrix) {
    matrix->used = 0;
}

int API_VIS lfr_builder_init (
    lfr_builder_t builder,
    size_t capacity
) {
    builder->capacity = capacity;
    builder->used = 0;

    builder->relations = calloc(capacity, sizeof(*builder->relations));    
    if (builder->relations == NULL) {
        lfr_builder_destroy(builder);
        return -ENOMEM;
    }
    
    return 0;
}

typedef struct {
    lfr_uniform_block_index_t block_positions[2];
    uint8_t keyout[2*LFR_BLOCKSIZE];
    uint64_t augmented;
} _lfr_hash_result_t;

static inline __attribute__((always_inline))
_lfr_hash_result_t _lfr_uniform_hash (
    const uint8_t *key,
    size_t key_length,
    lfr_salt_t salt,
    size_t nblocks
) {
    _lfr_hash_result_t result;
    size_t s = sizeof(result.keyout);
    s += (-s)%8;
    uint8_t hash[s];
    
    hash_result_t data = murmur3_x64_128_extended_seed(key, key_length, salt);
    _lfr_uniform_sample_block_positions(result.block_positions,nblocks,(uint32_t)data.high64,data.high64>>32);
    result.augmented = data.low64;

    // PERF: can we optimize this further eg by not cycling through ui2le?
    for (unsigned i=0; i<s/8; i++) {
        data.high64 += data.low64;
        data.low64  ^= rotl64(data.high64, 39);
        ui2le(&hash[i*8],  8, data.low64);
    }
    memcpy(result.keyout,hash,sizeof(result.keyout));

    return result;
}

/* A structure for tracking when a given half-row will meet its other half */
typedef struct {
    uint32_t row;       // When it gets merged, what's its row index?
    uint8_t merge_step; // which step will this row be merged on?
} resolution_t;

/* A group of half-rows */
typedef struct {
    tile_matrix_t data;
    resolution_t *row_resolution;
    tile_matrix_systematic_t systematic; // from parents
    size_t cols;
    size_t rows;
    lfr_salt_t salt;
    int error;
#if LFR_THREADED
    pthread_cond_t cv;
    pthread_mutex_t mut;
    uint32_t mark, solved;
#endif
} group_t;


static lfr_uniform_block_index_t resolution_block (
    lfr_uniform_block_index_t a,
    lfr_uniform_block_index_t b
) {
    /* Given a row spanning two blocks, in which block will it be used as one row? */
    assert(a<b);
    lfr_uniform_block_index_t delta = 1ull << high_bit(a ^ b);
    return b & -delta;
}

static void lfr_builder_destroy_groups(group_t *groups, size_t ngroups) {
    /* Destroy and free the groups, as part of a cleanup routine */
    if (groups) {
        for (size_t i=0; i<ngroups; i++) {
            tile_matrix_destroy(&groups[i].data);
            tile_matrix_systematic_destroy(&groups[i].systematic);
            free(groups[i].row_resolution);
#if LFR_THREADED
            pthread_cond_destroy(&groups[i].cv);
            pthread_mutex_destroy(&groups[i].mut);
#endif
        }
        free(groups);
    }
}

static int wait_for_solved(group_t *group, uint8_t solved) {
#if LFR_THREADED
    pthread_mutex_lock(&group->mut);
    while (group->solved != solved && !group->error) {
        pthread_cond_wait(&group->cv, &group->mut);
    }
    int ret = group->error;
    pthread_mutex_unlock(&group->mut);
    return ret;
#else
    (void)solved;
    return group->error;
#endif
}

static void mark_as_solved(group_t *group, uint8_t solved, int error) {
#if LFR_THREADED
    pthread_mutex_lock(&group->mut);
    group->solved = solved;
    group->error = error;
    pthread_cond_broadcast(&group->cv);
    pthread_mutex_unlock(&group->mut);
#else
    (void)solved;
    group->error = error;
#endif
}

static uint32_t mark_as_mine(group_t *group, uint32_t my_mark) {
#if LFR_THREADED
    // return 0 on success, 1 if already taken
    return __atomic_exchange_n(&group->mark, my_mark, __ATOMIC_RELAXED) == my_mark;
#else
    (void)group;
    (void)my_mark;
    return 0;
#endif
}

/** Return the number of blocks required for a given number of relations */
static inline size_t nblocks(size_t nrelns) {
    return _lfr_uniform_provision_columns(nrelns) / 8 / LFR_BLOCKSIZE;
}

/* Load the data into matrices for the group solver */
static int lfr_uniform_build_setup (
    group_t **pgroups,
    const lfr_builder_t builder,
    lfr_salt_t salt,
    unsigned value_bits
) {
    int ret=0;
    size_t blocks = nblocks(builder->used);
    size_t log_blocks = high_bit(blocks-1);
    size_t ngroups = 1ull << (2+log_blocks);
    group_t *groups = calloc(ngroups, sizeof(*groups));
    
    /* Count number of elements in each block. */
    for (size_t i=0; i<builder->used; i++) {
        _lfr_hash_result_t hash = _lfr_uniform_hash (
            builder->relations[i].query,
            builder->relations[i].query_length,
            salt, blocks
        );
        size_t a = 1+2*hash.block_positions[0];
        size_t b = 1+2*hash.block_positions[1];
        groups[a].rows++;
        groups[b].rows++;
    }

#if LFR_THREADED
    /* Set up the mutexes etc */
    for (size_t i=0; i<ngroups; i++) {
        ret = pthread_mutex_init(&groups[i].mut, NULL);
        if (ret) goto fail;
        ret = pthread_cond_init(&groups[i].cv, NULL);
        if (ret) goto fail;
        groups[i].solved = i&1; // odd ones are the initial ones
    }
#endif

    /* Create matrices */
    for (size_t i=0; i<blocks; i++) {
        group_t *g = &groups[2*i+1];
        g->cols = LFR_BLOCKSIZE*8;
        ret = tile_matrix_init(&g->data, g->rows, LFR_BLOCKSIZE*8, value_bits);
        if (ret) { goto fail; }
        g->row_resolution = calloc(g->rows, sizeof(*g->row_resolution));
        if (g->row_resolution == NULL) { goto fail; }
        g->rows = 0; // reset for next step
    }

    /* Success! */
    *pgroups = groups;
    return 0;

fail:
    if (ret==0) ret = -ENOMEM;
    lfr_builder_destroy_groups(groups, ngroups);
    return ret;
}

static int lfr_uniform_half_merge(
    tile_matrix_t *working,
    resolution_t *merged_resolution_data,
    group_t *half,
    size_t rows_expected,
    size_t offset,
    uint8_t merge_step
) {
    /* The merge step merges certain rows from each of two halves into a single matrix,
     * then row reduces it.  This subroutine does half the merge: it copies the rows
     * to be dealt with in this block to working, and limits the `half` matrix to the
     * rows not to be merged.
     */
    
    // Create scratch matrix
    tile_matrix_t copy_scratch[1];
    int ret = tile_matrix_init(copy_scratch,rows_expected,half->data.cols,half->data.aug_cols);
    if (ret) { return ret; }

    // Copy the desired rows into the scratch matrix
    size_t ncopied=0, n_not_copied=0;
    for (size_t row=0; row<half->data.rows; row++) {
        if (half->row_resolution[row].merge_step == merge_step) {
            ncopied++;
            size_t target_row = half->row_resolution[row].row;
            assert(target_row < rows_expected);
            tile_matrix_xor_row(copy_scratch,&half->data,target_row,row);
        } else {
            merged_resolution_data[n_not_copied] = half->row_resolution[row];
            // can't use the faster (?) xor here because the target isn't 0
            tile_matrix_copy_rows(&half->data,&half->data,n_not_copied,row,1);
            n_not_copied++;
        }
    }
    tile_matrix_change_nrows(&half->data, n_not_copied);
    assert(ncopied == rows_expected);

    // Copy the scratch matrix into the working matrix, at some offset
    tile_matrix_copy_cols(working,copy_scratch,offset,0,half->data.cols);
    tile_matrix_xor_augdata(working,copy_scratch);
    tile_matrix_destroy(copy_scratch);
    return 0;
}

static int lfr_uniform_project_out(
    tile_matrix_t *target,
    group_t *group,
    const tile_matrix_systematic_t *sys,
    size_t nrows,
    size_t ech_offset,
    size_t non_ech_offset,
    size_t n_ech
) {
    /* Half of the projection phase of the merge step.
    *
     * Take the left or right group in `group`.  The rows that are merged with
     * the other side are already accounted for in `sys`, and the unmerged ones remain.
     * This function deals with the unmerged rows, and is destructive to `group`
     */
    tile_matrix_t tmpb[1];
    memset(tmpb,0,sizeof(tmpb)); // preclear in case we fail before init'ing
    int ret = tile_matrix_init(target, nrows, sys->rhs.cols, sys->rhs.aug_cols);
    if (ret) { goto done; }

    // Copy the non-echelon columns to output, and the echelon ones to tmpb
    ret = tile_matrix_init(tmpb, nrows, n_ech, 0);
    if (ret) { goto done; }

    // Copy depending on whether in/out of echelon.
    // Profiling indicates that this is a performance-sensitive routine.  Instead
    // of copying the columns one-at-a-time, copy them in bulk
    size_t col_out_tmpb=0, col_out_target=non_ech_offset;
    size_t n_in_echelon=0, n_not_in_echelon=0, col;
    for (col=0; col<group->data.cols; col++) {
        if (bitset_test_bit(sys->column_is_in_echelon, col+ech_offset)) {
            if (n_not_in_echelon) {
                // previous group was not in echelon, but this is; resolve them
                tile_matrix_copy_cols(target,&group->data,col_out_target,col-n_not_in_echelon,n_not_in_echelon);
                col_out_target += n_not_in_echelon;
                n_not_in_echelon = 0;
            }
            n_in_echelon++;
        } else {
            if (n_in_echelon) {
                // previous group was in echelon, but this is not; resolve them
                tile_matrix_copy_cols(tmpb,&group->data,col_out_tmpb,col-n_in_echelon,n_in_echelon);
                col_out_tmpb += n_in_echelon;
                n_in_echelon = 0;
            }
            n_not_in_echelon++;
        }
    }
    if (n_not_in_echelon) {
        tile_matrix_copy_cols(target,&group->data,col_out_target,col-n_not_in_echelon,n_not_in_echelon);
        col_out_target += n_not_in_echelon;
    } else if (n_in_echelon) {
        tile_matrix_copy_cols(tmpb,&group->data,col_out_tmpb,col-n_in_echelon,n_in_echelon);
        col_out_tmpb += n_in_echelon;
    }
    assert(col_out_tmpb == tmpb->cols);
    tile_matrix_xor_augdata(target, &group->data);

    tile_matrix_t sys_submatrix[1]; // Not destroyed because it's a submatrix
    size_t prev_ech = ech_offset-non_ech_offset; // previous rows in echelon
    prev_ech += (-prev_ech)%TILE_SIZE; // padded
    _tile_aligned_submatrix(sys_submatrix, &sys->rhs, n_ech, prev_ech);

    // Do the projection
    tile_matrix_multiply_accumulate(target, tmpb, sys_submatrix);

done:
    tile_matrix_destroy(tmpb);
    if (ret) tile_matrix_destroy(target);
    return ret;
}

static int lfr_uniform_build_merge(group_t *result, group_t *left, group_t *right, uint8_t merge_step, int last) {
    /**
     * Given a collection of half rows in e.g. groups 1 and 3 (would be blocks 0 and 1 in orig matrix),
     * combine them into status for group 2 spanning both blocks.
     * 
     * To do this, note that some of the half rows merge to form full rows, and some do not.
     * The half rows forming full rows end up rearranged, as indexed by their resolution data.
     * Make a matrix of the full rows, and systematize it.  These rows are now accounted for.
     * 
     * Then take the L remaining rows of the left group, and the R remaining rows of the right
     * group, mod the systematic matrix.  This may increase or reduce their number of columns.
     * Put these in the resulting group.
     * 
     * For the remaining half rows, propagate the resolution data to the output.
     * 
     * Remember the systematic matrix and the output, but delete the two groups of
     * input half-rows.
     */
    int ret;

    // make a working matrix for the merged rows
    size_t augcols = left->data.aug_cols;
    assert(augcols == right->data.aug_cols);
    tile_matrix_t working[1];
    size_t nrows = result->rows;
    ret = tile_matrix_init(working, nrows, left->data.cols + right->data.cols, augcols);
    if (ret) { goto done; }

    // allocate the merged resolution data
    size_t n_resolution = left->data.rows + right->data.rows - 2*result->rows;
    resolution_t *merged_resolution = result->row_resolution = calloc(n_resolution, sizeof*merged_resolution);
    if (n_resolution > 0 && merged_resolution == NULL) {
        ret = -ENOMEM;
        goto done;
    }

    // Copy matrices into the working one
    ret = lfr_uniform_half_merge(working, merged_resolution, left,  result->rows, 0, merge_step);
    if (ret) goto done;

    merged_resolution += left->data.rows;
    ret = lfr_uniform_half_merge(working, merged_resolution, right, result->rows, left->data.cols, merge_step);
    if (ret) goto done;

    // Put the merged matrix in systematic form
    ret = tile_matrix_systematic_form(&result->systematic, working);
    tile_matrix_destroy(working);
    if (ret) goto done;

    // Align the "left" and "right" halves of the systematic form matrix
    size_t cur_rows = result->systematic.rhs.rows;
    size_t left_ech = bitset_popcount(result->systematic.column_is_in_echelon, left->data.cols);
    size_t left_non_ech = left->data.cols - left_ech;
    size_t right_ech = cur_rows - left_ech;
    size_t pad_rows = (-left_ech) % TILE_SIZE;
    if (pad_rows > 0 && left_ech < cur_rows) {
        tile_matrix_change_nrows(&result->systematic.rhs, cur_rows + pad_rows);
        tile_matrix_move_rows(&result->systematic.rhs, left_ech+pad_rows, left_ech, cur_rows-left_ech);
    }

    if (last) { goto done; } // there shouldn't be any rows left over anyway

    // Create the merged matrix
    //  ... project out left
    size_t merged_rows_left  = left ->data.rows;
    ret = lfr_uniform_project_out(&result->data, left, &result->systematic, merged_rows_left, 0, 0, left_ech);
    if (ret) { goto done; }

    //  ... project out right, into a temporary matrix
    size_t merged_rows_right = right->data.rows;
    tile_matrix_t merge_tmp[1];
    ret = lfr_uniform_project_out(merge_tmp, right, &result->systematic, merged_rows_right, left->data.cols, left_non_ech, right_ech);
    if (ret) { goto done; } // in this case lfr_uniform_project_out destroys merge_tmp

    // Append the temporary matrix to the bottom of the result
    ret = tile_matrix_change_nrows(&result->data, merged_rows_left + merged_rows_right);
    if (ret) {
        tile_matrix_destroy(merge_tmp);
        goto done;
    }
    // PERF: "copy_rows_unordered?"
    tile_matrix_copy_rows(&result->data, merge_tmp, merged_rows_left, 0, merged_rows_right);
    tile_matrix_destroy(merge_tmp);

    result->cols = result->systematic.rhs.cols;

done:
    free(left->row_resolution);
    left->row_resolution = NULL;
    free(right->row_resolution);
    right->row_resolution = NULL;
    tile_matrix_destroy(working);
    tile_matrix_destroy(&left->data);
    tile_matrix_destroy(&right->data);
    return ret;
}

static int lfr_uniform_move_group(group_t *target, group_t *source) {
    /* In the case that a group is unpaired (i.e. it's the last group),
     * it "merges" with nothing.  This result in an output that's the
     * same as the input.
     */

    // The systematic form does nothing
    int ret = tile_matrix_trivial_systematic_form(&target->systematic, source->cols);
    if (ret) return ret;

    // Move the row resolution
    target->row_resolution = source->row_resolution;
    source->row_resolution = NULL;
    target->cols = source->cols;

    // Move the matrix
    target->data = source->data;
    memset(&source->data, 0, sizeof(source->data));
    return 0;
}

static int lfr_uniform_backward_solve(group_t *left, group_t *right, group_t *center) {
    /* Backward solution step.
     * This is relatively easy: at each level we have an equation of the form 
     */
    size_t augcols = center->data.aug_cols;

    tile_matrix_t tmp[1];
    int ret = tile_matrix_init(tmp, center->systematic.rhs.rows, 0, augcols);
    if (ret) { goto done; }

    size_t rows_left = left->cols, rows_right = right->cols;
    ret = tile_matrix_init(&left->data,  rows_left, 0,  augcols);
    if (ret) { goto done; }

    ret = tile_matrix_init(&right->data, rows_right, 0, augcols);
    if (ret) { goto done; }

    // multiply up
    tile_matrix_multiply_accumulate(tmp, &center->systematic.rhs, &center->data);

    size_t col_test=0, sys_row=0, ipt_row=0;

    // unmerge left
    for (size_t row=0; row<rows_left; row++) {
        if (bitset_test_bit(center->systematic.column_is_in_echelon, col_test++)) {
            // pull it from systematic component.  Using xor because the input is zero
            tile_matrix_xor_row(&left->data, tmp, row, sys_row++);
        } else {
            // pull it from input
            tile_matrix_xor_row(&left->data, &center->data, row, ipt_row++);
        }
    }

    // Account for the padding in the sys matrix
    sys_row += (-sys_row) % TILE_SIZE;

    // unmerge right
    for (size_t row=0; row<rows_right; row++) {
        if (bitset_test_bit(center->systematic.column_is_in_echelon, col_test++)) {
            // pull it from systematic component
            tile_matrix_xor_row(&right->data, tmp, row, sys_row++);
        } else {
            // pull it from input
            tile_matrix_xor_row(&right->data, &center->data, row, ipt_row++);
        }
    }

done:
    tile_matrix_destroy(tmp);
    tile_matrix_destroy(&center->data);
    tile_matrix_systematic_destroy(&center->systematic);
    return ret;
}

typedef struct  {
    const lfr_builder_s *matrix; // TODO: rename
    group_t *groups;
    size_t ngroups;
    lfr_salt_t salt;
    unsigned value_bits;
#if LFR_THREADED
    pthread_mutex_t mut;
#endif
    int counter;
    int nthreads;
    int ret;
} lfr_uniform_build_args_t;

static void initialize_row (
    group_t *left,
    group_t *right,
    group_t *resolution,
    const uint8_t *keydata,
    const uint8_t *augdata,
    int merge_step
) {
#if LFR_THREADED
        pthread_mutex_lock(&left->mut);
        pthread_mutex_lock(&right->mut);
        pthread_mutex_lock(&resolution->mut);
#endif
        size_t row_left  = left->rows++, row_right = right->rows++;
        size_t row_res   = resolution->rows++;

        // PERF: this step is pretty slow
        tile_matrix_set_row(&left->data,  row_left,  keydata, NULL);
        left->row_resolution[row_left].merge_step = merge_step;
        left->row_resolution[row_left].row = row_res;

        tile_matrix_set_row(&right->data, row_right, &keydata[LFR_BLOCKSIZE], augdata);
        right->row_resolution[row_right].merge_step = merge_step;
        right->row_resolution[row_right].row = row_res;
#if LFR_THREADED
        pthread_mutex_unlock(&resolution->mut);
        pthread_mutex_unlock(&left->mut);
        pthread_mutex_unlock(&right->mut);
#endif
}

static void *lfr_uniform_build_thread (void *args_void) {
    lfr_uniform_build_args_t *args = (lfr_uniform_build_args_t *)args_void;
    const lfr_builder_s *builder = args->matrix;
    group_t *groups = args->groups;
    size_t ngroups = args->ngroups;
    
    // PERF: the bottleneck is waiting for the one thread to solve the last
    // (or for more threads, maybe nth-last) matrix
    // So, to be really effective in threaded mode, we need a multithreaded
    // version of he matrix solver

    // Grab a thread ID
#if LFR_THREADED
    pthread_mutex_lock(&args->mut);
    int threadid = args->counter++, nthreads = args->nthreads;
    pthread_mutex_unlock(&args->mut);
#else
    int threadid = 0, nthreads = 1;
#endif

    /* Copy rows into submatrices, and count resolutions */
    size_t start = args->matrix->used*threadid / nthreads;
    size_t end = args->matrix->used*(threadid+1) / nthreads;
    size_t blocks = nblocks(args->matrix->used);
    for (size_t i=start; i<end; i++) {
        _lfr_hash_result_t hash = _lfr_uniform_hash(
            builder->relations[i].query,
            builder->relations[i].query_length,
            args->salt,
            blocks
        );
        hash.augmented ^= builder->relations[i].response;
        lfr_uniform_block_index_t block_left  = 2 * hash.block_positions[0] + 1;
        lfr_uniform_block_index_t block_right = 2 * hash.block_positions[1] + 1;

        if (block_left > block_right) {
            lfr_uniform_block_index_t tmp = block_left;
            block_left = block_right;
            block_right = tmp;

            uint8_t tmpb[LFR_BLOCKSIZE];
            memcpy(tmpb,hash.keyout,LFR_BLOCKSIZE);
            memcpy(hash.keyout,&hash.keyout[LFR_BLOCKSIZE],LFR_BLOCKSIZE);
            memcpy(&hash.keyout[LFR_BLOCKSIZE],tmpb,LFR_BLOCKSIZE);
        }


        uint32_t resolution = resolution_block(block_left, block_right);
        uint32_t merge_step = __builtin_ctzll(resolution);
        uint8_t augmented_b[sizeof(hash.augmented)];
        ui2le(augmented_b, sizeof(augmented_b), hash.augmented);

        initialize_row(&groups[block_left], &groups[block_right], &groups[resolution], hash.keyout, augmented_b, merge_step);
    }

    // synchronize
    wait_for_solved(&groups[0],threadid);
    mark_as_solved(&groups[0],threadid+1,0);
    wait_for_solved(&groups[0],nthreads);
    
    int lgstep, ret, i_did_last = 0;
    for (lgstep=1; 1ull<<lgstep < ngroups; lgstep++) {
        // check in to see if we failed
#if LFR_THREADED
        pthread_mutex_lock(&args->mut);
        ret = args->ret;
        pthread_mutex_unlock(&args->mut);
        if (ret) return NULL;
#endif
            
        size_t step = 1ull << lgstep;
        int last = 2*step >= ngroups;
        for (size_t mid=step; mid<ngroups; mid += 2*step) {
            uint64_t left_i = mid-step/2, right_i = mid+step/2;
            group_t *out = &groups[mid], *left = &groups[left_i], *right = &groups[right_i];
            
            if (mark_as_mine(out,1)) continue;
            ret = wait_for_solved(left,1);
            if (!ret) ret = wait_for_solved(right,1);
            
            if (ret) {
                // fall through
            } else if (right->data.rows == 0) {
                ret = lfr_uniform_move_group(out, left);
            } else if (left->data.rows == 0) {
                ret = lfr_uniform_move_group(out, right);
            } else {
                ret = lfr_uniform_build_merge(&groups[mid], left, right, lgstep, last);
            }
            mark_as_solved(out,1,ret); // don't die and leave them hanging
        
            if (ret) goto done;
            if (last) i_did_last = 1;
        }
    }
    
    // Start the backprop with remaining free variables all set to 0
    lgstep--;
    group_t *final_group = &groups[1ull<<lgstep];
    if (i_did_last) {
        ret = tile_matrix_init(&final_group->data, final_group->systematic.rhs.cols, 0, args->value_bits);
        mark_as_solved(final_group,2,ret);
    } else {
        ret = wait_for_solved(final_group,2);
    }
    if (ret) goto done;

    // backward solve, going back down the tree
    for (; lgstep >= 1; lgstep--) {
        size_t step = 1ull << lgstep;
        for (size_t mid=step; mid<ngroups; mid += 2*step) {
            uint64_t left_i = mid-step/2, right_i = mid+step/2;
            group_t *in = &groups[mid], *left = &groups[left_i], *right = &groups[right_i];

            if (mark_as_mine(in,2)) continue;
            ret = wait_for_solved(in,2);

            if (!ret) ret = lfr_uniform_backward_solve(left, right, in);
            mark_as_solved(left,2,ret);
            mark_as_solved(right,2,ret);
            if (ret) goto done;
        }
    }
    
done:
#if LFR_THREADED
    if (ret) {
        pthread_mutex_lock(&args->mut);
        args->ret = ret;
        pthread_mutex_unlock(&args->mut);
    }
#else
    args->ret = ret;
#endif
    return NULL;
}

void API_VIS lfr_uniform_map_destroy(lfr_uniform_map_t doomed) {
    if (doomed->data_is_mine) free(doomed->data);
    memset(doomed, 0, sizeof(*doomed));
}

int API_VIS lfr_uniform_build (lfr_uniform_map_t output, const lfr_builder_t builder, unsigned value_bits, lfr_salt_t salt) {
    return lfr_uniform_build_threaded(output,builder,value_bits,salt,0);
}

size_t API_VIS lfr_uniform_map_size(const lfr_uniform_map_t map) {
    return map->blocks * LFR_BLOCKSIZE * map->value_bits;
}

int API_VIS lfr_uniform_build_threaded (
    lfr_uniform_map_t output,
    const lfr_builder_t builder,
    unsigned value_bits,
    lfr_salt_t salt,
    int nthreads
) {
    int ret=0;
    size_t blocks = nblocks(builder->used);
    size_t ngroups = 1ull << (2+high_bit(blocks-1));
    group_t *groups = NULL;
    memset(output,0,sizeof(*output));

    if (value_bits > 8*sizeof(lfr_response_t)) return -EINVAL;

#if LFR_THREADED
    size_t len = sizeof(nthreads);
    int mib[2] = { CTL_HW, HW_NCPU }, sret=0;
    if (nthreads <= 0) sret = sysctl(mib, 2, &nthreads, &len, NULL, 0);
    if (nthreads <= 0 || sret != 0) nthreads = 1;
    pthread_t threads[nthreads];
#else
    nthreads = 1;
#endif

    lfr_uniform_build_args_t args;
    memset(&args,0,sizeof(args));
    args.salt = salt;
    ret = lfr_uniform_build_setup(&groups, builder, salt, value_bits);
    if (ret) { goto done; }

    // Forward solve
    args.matrix = &builder[0];
    args.value_bits = value_bits;
    args.groups = groups;
    args.ngroups = ngroups;
    args.counter = 0;
    args.nthreads = nthreads;
#if LFR_THREADED
    ret = pthread_mutex_init(&args.mut, NULL);
    if (ret) { goto done; }
#endif
    args.ret = 0;
    
#if LFR_THREADED
    // Launch the solve threads
    int i;
    for (i=1; i<nthreads; i++) {
        ret = pthread_create(&threads[i], NULL, lfr_uniform_build_thread, &args);
        if (ret) break;
    }
#endif
    // grab a thread myself
    lfr_uniform_build_thread((void*) &args);
    
#if LFR_THREADED
    // Collect them
    for (int j=1; j<=i && j<nthreads; j++) {
        pthread_join(threads[j], NULL);
    }
    pthread_mutex_destroy(&args.mut);
#endif
    if (!ret) ret = args.ret;
    if (ret) goto done;

    // Write output
    output->data = calloc(value_bits, blocks * LFR_BLOCKSIZE);
    if (output->data == NULL) {
        ret = -ENOMEM;
        goto done;
    }
    output->salt = salt;
    output->value_bits = value_bits;
    output->data_is_mine = 1;
    output->blocks = blocks;

    size_t byte_index=0;
    for (size_t block=0; block<blocks; block++) {
        const tile_matrix_t *m = &groups[2*block+1].data;
        size_t tstride = m->stride, off = TILES_SPANNING(m->cols);
        for (size_t which_augcol=0; which_augcol<value_bits; which_augcol++) {
            for (size_t tile=0; tile<LFR_BLOCKSIZE*8/TILE_SIZE; tile++) {
                tile_t out = m->data[tile*tstride + off + which_augcol/TILE_SIZE] >> (TILE_SIZE*(which_augcol % TILE_SIZE));
                for (int b=0; b<TILE_SIZE/8; b++) {
                    output->data[byte_index++] = (uint8_t)(out>>(8*b));
                }
            }
        }
    }

done:
    lfr_builder_destroy_groups(groups, ngroups);
    if (ret != 0 && ret != -ENOMEM) ret = -EAGAIN;
    return ret;
}

int API_VIS lfr_builder_insert (
    lfr_builder_t builder,
    const uint8_t *key,
    size_t keybytes,
    uint64_t value
) {
    if (builder->used >= builder->capacity) return -EINVAL; // TODO: resize??
    size_t row = builder->used++;
    builder->relations[row].query = key;
    builder->relations[row].query_length = keybytes;
    builder->relations[row].response = value;
    return 0;
}

typedef struct {
    lfr_uniform_block_t x;
} __attribute__((packed)) unaligned_block_t;

uint64_t API_VIS lfr_uniform_query (
    const lfr_uniform_map_t map,
    const uint8_t *key,
    size_t keybytes
) {
    size_t value_bits = map->value_bits;
    
    _lfr_hash_result_t hash = _lfr_uniform_hash(key, keybytes, map->salt, map->blocks);
    lfr_uniform_block_t key_blk[2];
    memcpy(key_blk, hash.keyout, sizeof(key_blk));
    uint64_t ret = hash.augmented;
    uint64_t mask;
    if (value_bits == 8*sizeof(ret)) {
        mask = -1ull;
    } else {
        mask = (1ull<<value_bits) - 1;
    }

    const unaligned_block_t *vectors_blk = (const unaligned_block_t *) map->data;
    const unaligned_block_t *blkptr0 = &vectors_blk[value_bits*hash.block_positions[0]];
    const unaligned_block_t *blkptr1 = &vectors_blk[value_bits*hash.block_positions[1]];

    #pragma clang loop vectorize(disable) // small trip count, not worth it
    for (size_t obit=0; obit<value_bits; obit++) {
        lfr_uniform_block_t dot = (blkptr0[obit].x & key_blk[0]) ^ (blkptr1[obit].x & key_blk[1]);
        ret ^= (uint64_t)parity(dot) << obit;
    }
    return ret & mask;
}
