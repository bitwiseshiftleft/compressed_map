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

// Internal: sample the block positions for a matrix
extern _lfr_uniform_row_indices_s _lfr_uniform_sample_block_positions (
    size_t nblocks,
    const uint8_t *sample_bytes // size is LFR_META_SAMPLE_BYTES
);

void API_VIS lfr_uniform_builder_destroy(lfr_uniform_builder_t matrix) {
    free(matrix->row_meta);
    free(matrix->data);
    memset(matrix,0,sizeof(*matrix));
}

void API_VIS lfr_uniform_builder_reset(lfr_uniform_builder_t matrix) {
    matrix->used = 0;
}

static inline size_t lfr_uniform_stride(const lfr_uniform_builder_t matrix) {
    return 2*LFR_BLOCKSIZE + BYTES(matrix->value_bits);
}

int API_VIS lfr_uniform_builder_init (
    lfr_uniform_builder_t matrix,
    size_t capacity,
    size_t value_bits,
    lfr_uniform_salt_t salt
) {
    matrix->capacity = capacity;
    matrix->blocks = _lfr_uniform_provision_columns(capacity) / 8 / LFR_BLOCKSIZE;
    matrix->value_bits = value_bits;
    matrix->salt = salt;
    matrix->used = 0;
    
    size_t stride = lfr_uniform_stride(matrix);
    matrix->row_meta = malloc(capacity * sizeof(*matrix->row_meta));
    matrix->data = calloc(capacity, stride);
        
    if (matrix->row_meta == NULL || matrix->data == NULL) {
        lfr_uniform_builder_destroy(matrix);
        return -ENOMEM;
    }
    
    return 0;
}

static uint8_t *lfr_uniform_row(const lfr_uniform_builder_t matrix, size_t row) {
    if (row > matrix->used) return NULL;
    return &matrix->data[lfr_uniform_stride(matrix)*row];
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
    lfr_uniform_salt_t salt;
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

static void lfr_uniform_builder_destroy_groups(group_t *groups, size_t ngroups) {
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

/* Load the data into matrices for the group solver */
static int lfr_uniform_build_setup (
    group_t **pgroups,
    const lfr_uniform_builder_t matrix
) {
    int ret=0;
    size_t log_blocks = high_bit(matrix->blocks-1);
    size_t ngroups = 1ull << (2+log_blocks);
    group_t *groups = calloc(ngroups, sizeof(*groups));
    
    /* Count number of elements in each block. */
    for (size_t i=0; i<matrix->used; i++) {
        size_t a = 1+2*matrix->row_meta[i].blocks[0];
        size_t b = 1+2*matrix->row_meta[i].blocks[1];
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
    for (size_t i=0; i<matrix->blocks; i++) {
        group_t *g = &groups[2*i+1];
        g->cols = LFR_BLOCKSIZE*8;
        ret = tile_matrix_init(&g->data, g->rows, LFR_BLOCKSIZE*8, matrix->value_bits);
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
    lfr_uniform_builder_destroy_groups(groups, ngroups);
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
    const lfr_uniform_builder_s *matrix;
    group_t *groups;
    size_t ngroups;
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
    const uint8_t *row_data,
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
        tile_matrix_set_row(&left->data,  row_left,  row_data, NULL);
        left->row_resolution[row_left].merge_step = merge_step;
        left->row_resolution[row_left].row = row_res;

        tile_matrix_set_row(&right->data, row_right, &row_data[LFR_BLOCKSIZE], &row_data[2*LFR_BLOCKSIZE]);
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
    for (size_t i=start; i<end; i++) {
        const uint8_t *row_data = lfr_uniform_row(args->matrix, i);
        lfr_uniform_block_index_t block_left  = 2 * args->matrix->row_meta[i].blocks[0] + 1;
        lfr_uniform_block_index_t block_right = 2 * args->matrix->row_meta[i].blocks[1] + 1;

        uint32_t resolution = resolution_block(block_left, block_right);
        uint32_t merge_step = __builtin_ctzll(resolution);

        initialize_row(&groups[block_left], &groups[block_right], &groups[resolution], row_data, merge_step);
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
        ret = tile_matrix_init(&final_group->data, final_group->systematic.rhs.cols, 0, args->matrix->value_bits);
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

int API_VIS lfr_uniform_build (lfr_uniform_map_t output, const lfr_uniform_builder_t builder) {
    return lfr_uniform_build_threaded(output,builder,0);
}

size_t API_VIS lfr_uniform_builder_size(const lfr_uniform_builder_t builder) {
    return builder->blocks * LFR_BLOCKSIZE * builder->value_bits;
}

size_t API_VIS lfr_uniform_map_size(const lfr_uniform_map_t map) {
    return map->blocks * LFR_BLOCKSIZE * map->value_bits;
}

int API_VIS lfr_uniform_build_threaded (lfr_uniform_map_t output, const lfr_uniform_builder_t matrix, int nthreads) {
    int ret=0;
    size_t ngroups = 1ull << (2+high_bit(matrix->blocks-1));
    group_t *groups = NULL;
    memset(output,0,sizeof(*output));

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
    ret = lfr_uniform_build_setup(&groups, matrix);
    if (ret) { goto done; }

    // Forward solve
    args.matrix = &matrix[0];
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
        ret = pthread_init(&threads[i], NULL, lfr_uniform_build_thread, &args);
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
    output->data = calloc(matrix->value_bits, matrix->blocks * LFR_BLOCKSIZE);
    if (output->data == NULL) {
        ret = -ENOMEM;
        goto done;
    }
    output->salt = matrix->salt;
    output->value_bits = matrix->value_bits;
    output->data_is_mine = 1;
    output->blocks = matrix->blocks;

    size_t byte_index=0;
    for (size_t block=0; block<matrix->blocks; block++) {
        const tile_matrix_t *m = &groups[2*block+1].data;
        size_t tstride = m->stride, off = TILES_SPANNING(m->cols);
        for (size_t which_augcol=0; which_augcol<matrix->value_bits; which_augcol++) {
            for (size_t tile=0; tile<LFR_BLOCKSIZE*8/TILE_SIZE; tile++) {
                tile_t out = m->data[tile*tstride + off + which_augcol/TILE_SIZE] >> (TILE_SIZE*(which_augcol % TILE_SIZE));
                for (int b=0; b<TILE_SIZE/8; b++) {
                    output->data[byte_index++] = (uint8_t)(out>>(8*b));
                }
            }
        }
    }

done:
    lfr_uniform_builder_destroy_groups(groups, ngroups);
    if (ret != 0 && ret != -ENOMEM) ret = -EAGAIN;
    return ret;
}

static inline __attribute__((always_inline)) void _lfr_uniform_hash (
    _lfr_uniform_row_indices_s *meta,
    uint8_t *keyout,
    uint8_t *affine_offset,
    size_t offset_bits,
    const uint8_t *key,
    size_t key_length,
    lfr_uniform_salt_t salt,
    size_t nblocks
) {
    size_t offset_bytes = BYTES(offset_bits);
    size_t s = LFR_META_SAMPLE_BYTES + 2*LFR_BLOCKSIZE + offset_bytes;
    s += (-s)%16;
    uint8_t hash[s];
    
    hash_result_t data = murmur3_x64_128_extended_seed(key, key_length, salt);
    ui2le(&hash[0], 8, data.low64);
    ui2le(&hash[8], 8, data.high64);
    *meta = _lfr_uniform_sample_block_positions(nblocks,hash);

    for (unsigned i=1; i<s/16; i++) {
        salt = fmix64(salt);
        hash_result_t data = murmur3_x64_128_extended_seed(key, key_length, salt);
        ui2le(&hash[i*16],  8, data.low64);
        ui2le(&hash[i*16+8],8, data.high64);
    }
    memcpy(keyout,&hash[LFR_META_SAMPLE_BYTES],2*LFR_BLOCKSIZE);
    memcpy(affine_offset,&hash[LFR_META_SAMPLE_BYTES+2*LFR_BLOCKSIZE],offset_bytes);
    if (offset_bits%8) {
        affine_offset[offset_bytes-1] &= (1<<(offset_bits%8))-1;
    }
}

int API_VIS lfr_uniform_insert (
    lfr_uniform_builder_t matrix,
    const uint8_t *key,
    size_t keybytes,
    uint64_t value
) {
    if (matrix->used >= matrix->capacity) return -EINVAL;
    size_t row = matrix->used++;
    size_t stride = lfr_uniform_stride(matrix);
    uint8_t *augdata = &matrix->data[stride*row+2*LFR_BLOCKSIZE];
    uint8_t *keydata = &matrix->data[stride*row];
    _lfr_uniform_hash (
        &matrix->row_meta[row],
        keydata,
        augdata,
        matrix->value_bits,
        key, keybytes, matrix->salt, matrix->blocks
    );
    if (matrix->row_meta[row].blocks[0] > matrix->row_meta[row].blocks[1]) {
        // swap
        lfr_uniform_block_index_t tmp = matrix->row_meta[row].blocks[0];
        matrix->row_meta[row].blocks[0] = matrix->row_meta[row].blocks[1];
        matrix->row_meta[row].blocks[1] = tmp;
        uint8_t tmpkey[LFR_BLOCKSIZE];
        memcpy(tmpkey, keydata, LFR_BLOCKSIZE);
        memcpy(keydata, &keydata[LFR_BLOCKSIZE], LFR_BLOCKSIZE);
        memcpy(&keydata[LFR_BLOCKSIZE], tmpkey, LFR_BLOCKSIZE);
    }
    for (size_t i=0; i<BYTES(matrix->value_bits); i++) {
        augdata[i] ^= value>>(8*i);
    }
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
    uint64_t ret;
    uint8_t value[sizeof(ret)];
    size_t value_bits = map->value_bits;
    _lfr_uniform_row_indices_s meta;
    lfr_uniform_block_t key_blk[2];
    
    // PERF: so the perf of this sucks  compared to hashing externally.
    // Why?  For two reasons: One, there's a lot of branching on keybytes in _lfr_uniform_hash.
    // Two, if you hash all the keys first (unrealistic!) then the CPU can probably prefetch
    // the memory, since the address is just sitting there.  Here the hash exceeds `the OOO window
    _lfr_uniform_hash(&meta, (uint8_t*)key_blk, value, 8*sizeof(value), key, keybytes, map->salt, map->blocks);

    ret = le2ui(value,sizeof(value)) & ((1ull << value_bits)-1);
    const unaligned_block_t *vectors_blk = (const unaligned_block_t *) map->data;
    const unaligned_block_t *blkptr0 = &vectors_blk[value_bits*meta.blocks[0]];
    const unaligned_block_t *blkptr1 = &vectors_blk[value_bits*meta.blocks[1]];

    #pragma clang loop vectorize(disable) // small trip count, not worth it
    for (size_t obit=0; obit<value_bits; obit++) {
        lfr_uniform_block_t dot = (blkptr0[obit].x & key_blk[0]) ^ (blkptr1[obit].x & key_blk[1]);
        ret ^= (uint64_t)parity(dot) << obit;
    }

    return ret;
}
