/*
 * @file matrix.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Matrix operations implemented using 16x16 "tile" submatrices.
 */

use crate::tilematrix::tile::{Tile,Edge,Index,Permutation,row_mul,row_mul_acc,
    bulk_swap_rows,bulk_swap2_rows,PERMUTE_ALL_ZERO};
use std::cmp::{min,max};
use std::ops::{AddAssign};
use rand::{Rng,thread_rng};
use crate::tilematrix::bitset::BitSet;

/** Return the number of tiles required to represent n rows or columns, rounded up */
fn tiles_spanning(n:usize) -> usize {
    let ebits = Tile::EDGE_BITS as usize;
    (n+ebits-1) / ebits
}

/** GF(2) matrix */
#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Debug)]
pub struct Matrix {
    pub rows      : usize,
    pub cols_main : usize,
    pub cols_aug  : usize,
    pub stride    : usize, // row-to-row stride measured in tiles
    pub tiles     : Vec<Tile>
}

impl Matrix {
    /** Create a new matrix. */
    pub fn new(rows:usize, cols_main:usize, cols_aug:usize) -> Matrix {
        let stride = tiles_spanning(cols_main) + tiles_spanning(cols_aug);
        let height = tiles_spanning(rows);
        let tiles  = vec!(Tile::ZERO; stride.checked_mul(height).unwrap());
        Matrix { rows, cols_main, cols_aug, stride, tiles }
    }

    /** Clears self and deallocates its memory */
    pub fn clear(&mut self) {
        self.rows = 0;
        self.cols_main = 0;
        self.cols_aug = 0;
        self.stride = 0;
        self.tiles.clear();
        self.tiles.shrink_to_fit();
    }

    /** Add a row as a little-endian collection of bytes */
    pub fn mut_add_row_as_bytes(&mut self, bytes_main:&[u8], bytes_aug:&[u8]) {
        const EDGE_BYTES : usize = Tile::EDGE_BITS / 8;
        debug_assert!(bytes_main.len() >=  (self.cols_main+7)/8);
        debug_assert!(bytes_aug.len()  >=  (self.cols_aug +7)/8);
        debug_assert!(bytes_aug.len()  % EDGE_BYTES == 0);
        debug_assert!(bytes_main.len() % EDGE_BYTES == 0);

        let row = self.rows;
        if self.rows % Tile::EDGE_BITS == 0 {
            self.tiles.resize(self.tiles.len() + self.stride, Tile::ZERO);
        }
        self.rows += 1;

        let trow = row/Tile::EDGE_BITS;
        let row_within = row % Tile::EDGE_BITS;
        let tcols = tiles_spanning(self.cols_main);
        let taugs = tiles_spanning(self.cols_aug);

        /* Set the cols */
        for tcol in 0..tcols {
            let mut e : Edge = 0;
            for b in 0..EDGE_BYTES {
                let idx = b+tcol*EDGE_BYTES;
                e |= (bytes_main[idx] as Edge) << (8*b);
            }
            self.tiles[trow*self.stride + tcol].mut_set_row(e, row_within);
        }

        /* Set the augs */
        for taug in 0..taugs {
            let mut e : Edge = 0;
            for b in 0..EDGE_BYTES {
                let idx = b+taug*EDGE_BYTES;
                e |= (bytes_aug[idx] as Edge) << (8*b);
            }
            self.tiles[trow*self.stride + tcols + taug].mut_set_row(e, row_within);
        }
    }

    /** Set all elements of the matrix to zero */
    #[allow(dead_code)]
    pub fn zeroize(&mut self) { self.tiles.fill(Tile::ZERO); }

    /** self += other */
    pub fn add_assign(&mut self, other: &Matrix) {
        let trows = tiles_spanning(min(self.rows,other.rows));
        let taugs = tiles_spanning(min(self.cols_aug,other.cols_aug));
        let off_self  = tiles_spanning(self.cols_main);
        let off_other = tiles_spanning(other.cols_main);
        let tcols = min(off_self, off_other);
        for trow in 0..trows {
            for tcol in 0..tcols {
                self.tiles[trow*self.stride+tcol] +=
                    other.tiles[trow*other.stride+tcol];
            }
            for taug in 0..taugs {
                self.tiles[trow*self.stride+off_self+taug] +=
                    other.tiles[trow*other.stride+off_other+taug];
            }
        }
    }

    /** Reserve memory for extra rows */
    pub fn reserve_rows(&mut self, additional:usize) {
        self.tiles.reserve(self.stride * tiles_spanning(additional));
    }

    /** Return a single bit of the matrix */
    #[allow(dead_code)]
    pub fn get_bit(&self, row:usize, col:usize) -> bool {
        let ebits = Tile::EDGE_BITS;
        self.tiles[(row/ebits)*self.stride + (col/ebits)].get_bit(row%ebits, col%ebits)
    }

    /** Return a single bit of the augmented component of the matrix */
    pub fn get_aug_bit(&self, row:usize, aug:usize) -> bool {
        let ebits = Tile::EDGE_BITS;
        self.tiles[(row/ebits)*self.stride + tiles_spanning(self.cols_main) + (aug/ebits)].get_bit(row%ebits, aug%ebits)
    }

    /** Print the matrix, for debugging purposes */
    #[allow(dead_code)]
    pub fn print(&self, name:&str) {
        let ebits = Tile::EDGE_BITS;
        print!("Matrix {}:\n", name);
        let tcols = tiles_spanning(self.cols_main);
        for i in 0..self.rows /*+ (self.rows.wrapping_neg() % ebits)*/ {
            // if i == self.rows { println!("----") }
            for j in 0..self.cols_main {
                let tile = self.tiles[(i/ebits)*self.stride + (j/ebits)];
                print!("{}", tile.get_bit((i%ebits) as Index, (j%ebits) as Index) as i8);
            }
            if self.cols_aug > 0 {
                print!(" | ");
                for j in 0..self.cols_aug {
                    let tile = self.tiles[(i/ebits)*self.stride + (j/ebits) + tcols];
                    print!("{}", tile.get_bit((i%ebits) as Index, (j%ebits) as Index) as i8);
                }

            }
            print!("\n");
        }
        print!("\n");
    }

    /** Self = a*b.  The dimensions should match; if they do not:
     *     The minimum number of rows, columns and augmentation will be used.
     *     The augmentation on b will be ignored if the column counts don't match.
     */
    pub fn assign_mul(&mut self, a: &Matrix, b: &Matrix) {
        self.zeroize();
        self.accum_mul(a,b);
    }

    /** Self += a*b.  The dimensions should match; if they do not:
     *     The minimum number of rows, columns and augmentation will be used.
     *     The augmentation on b will be ignored if the column counts don't match.
     */
    pub fn accum_mul(&mut self, a: &Matrix, b: &Matrix) {
        let trows    = tiles_spanning(min(self.rows,a.rows));
        let mut taug = tiles_spanning(min(self.cols_aug,b.cols_aug));
        let taug_a   = tiles_spanning(min(self.cols_aug,a.cols_aug));
        let tcself   = tiles_spanning(self.cols_main);
        let tcb      = tiles_spanning(b.cols_main);
        let tca      = tiles_spanning(a.cols_main);
        let tcols = min(tcself,tcb);
        if tcself != tcb {
            debug_assert!(taug==0);
            taug = 0;
        }
        let tcols_total = tcols + taug;
        let tjoin = tiles_spanning(min(a.cols_main,b.rows));

        for trow in 0..trows {
            let offset = trow*self.stride;
            let self_ptr = &mut self.tiles[offset .. offset+tcols_total];
            for tj in 0..tjoin {
                let a_tile =  a.tiles[trow*a.stride+tj+0];
                let b_ptr = &b.tiles[tj*b.stride .. tj*b.stride+tcols_total];
                row_mul_acc(self_ptr,a_tile,b_ptr);
            }
            for aug in 0..taug_a {
                self.tiles[offset+tcself+aug] ^= a.tiles[trow*a.stride+tca+aug];
            }
        }
    }

    /** Multiply self by another matrix, and return the result */
    pub fn mul(&self, b:&Matrix) -> Matrix {
        let mut result = Matrix::new(self.rows, b.cols_main, max(self.cols_aug, b.cols_aug));
        result.assign_mul(self,b);
        result
    }

    /** Randomize the matrix, for testing purposes */
    #[allow(dead_code)]
    pub fn randomize(&mut self) {
        let trows = tiles_spanning(self.rows);
        let tcols = tiles_spanning(self.cols_main);
        let taugs = tiles_spanning(self.cols_aug);
        let ebits = Tile::EDGE_BITS;
        let last_col_mask = if self.cols_main%ebits != 0 {
            Tile::cols_mask(0,self.cols_main%ebits)
        } else {
            Tile::FULL
        };
        let last_aug_mask = if self.cols_aug%ebits != 0 {
            Tile::cols_mask(0,self.cols_aug%ebits)
        } else {
            Tile::FULL
        };
        for trow in 0..trows {
            let row_mask = if trow==trows-1 && self.rows%ebits != 0 {
                Tile::rows_mask(0,self.rows%ebits)
            } else {
                Tile::FULL
            };
            for tcol in 0..tcols {
                let col_mask = if tcol==tcols-1 { last_col_mask } else { Tile::FULL };
                self.tiles[self.stride*trow+tcol] = thread_rng().gen::<Tile>() & row_mask & col_mask;
            }
            for taug in 0..taugs {
                let aug_mask = if taug==taugs-1 { last_aug_mask } else { Tile::FULL };
                self.tiles[self.stride*trow+tcols+taug] = thread_rng().gen::<Tile>() & row_mask & aug_mask;
            }
        }
    }

    /** Return mutable aliases to two different rows of self.
     * Must be different to satisfy the borrow checker.
     */
    #[inline(always)]
    fn two_rows(&mut self, r1:usize, r2:usize, start:usize) -> (&mut [Tile], &mut [Tile]) {
        let ttotal = tiles_spanning(self.cols_main) + tiles_spanning(self.cols_aug);
        assert!(start < ttotal);
        let stride = self.stride;
        let (r1,r2) = if r1 < r2 {
            let (lo,hi) = self.tiles.split_at_mut(r2*stride);
            (&mut lo[r1*stride..], hi)
        } else if r2 < r1 {
            let (lo,hi) = self.tiles.split_at_mut(r1*stride);
            (hi, &mut lo[r2*stride..])
        } else {
            panic!("two_rows must be disjoint!");
        };
        (&mut r1[start..ttotal], &mut r2[start..ttotal])
    }

    /** Swap rows ra..ra+nrows with rb..rb+nrows.  The ranges must be disjoint. */
    fn swap_rows(&mut self, ra:usize, rb:usize, mut nrows:usize, start:usize) {
        if ra == rb { return };
        let (mut ra, mut rb) = (min(ra,rb), max(ra,rb));
        const EBITS : usize = Tile::EDGE_BITS;
        let ttotal = tiles_spanning(self.cols_main) + tiles_spanning(self.cols_aug);
        while nrows > 0 {
            let (ra0,ra1) = (ra % EBITS, ra / EBITS);
            let (rb0,rb1) = (rb % EBITS, rb / EBITS);
            let cando = min(min(min(EBITS-ra0,EBITS-rb0),nrows),rb-ra);
            if ra1==rb1 {
                bulk_swap_rows(&mut self.tiles[ra1*self.stride+start..ra1*self.stride+ttotal],ra0,rb0,cando);
            } else {
                let (rowa,rowb) = self.two_rows(ra1,rb1,start);
                bulk_swap2_rows(rowa,rowb,ra0,rb0,cando);
            }
            ra += cando;
            rb += cando;
            nrows -= cando;
        }
    }

    /** Tile-aligned row operation: src_row[start..end] += tile * target_row[start..end]
     * src_row and target_row must differ.
     */
    fn rowop(&mut self, tile:Tile, target_row:usize, src_row:usize, start:usize) {
        let (src_sec,target_sec) = self.two_rows(src_row,target_row,start);
        row_mul_acc(target_sec, tile, src_sec);
    }

    /**
     * Put self in row-reduced echelon form.
     * Return the rank and a set of which columns are in echelon
     */
    pub fn rref(&mut self) -> (usize, BitSet) {
        let rows = self.rows;
        let cols = self.cols_main;
        let trows = tiles_spanning(rows);
        let tcols = tiles_spanning(cols);
        let tstride = self.stride;
        let ttotal = tcols + tiles_spanning(self.cols_aug);
        let mut column_is_in_echelon = BitSet::with_capacity(cols);
        let mut rank = 0;
        if trows == 0 { return (rank,column_is_in_echelon); } // trivial
    
        /* Put the tile-columns into echelon form, one after another */
        for tcol in 0..tcols {
            /* Which columns have we echelonized in this loop? */
            let mut ech : Edge = 0;
            let mut perm_matrix_cumulative = PERMUTE_ALL_ZERO;
    
            /* We want to echelonize Tile::EDGE_BITS rows if possible.  To do this, we select as
             * the active tile-row one which has Tile::EDGE_BITS rows not in the current echelon
             * structure, if possible.  Since the rows in echelon get moved to the beginning,
             * this is either trow_min or trow_min + 1.
             */
            let trow_min = rank/Tile::EDGE_BITS;
            let mut trow_begin = trow_min + (rank % Tile::EDGE_BITS != 0) as usize;
            if trow_min   >= trows { break; } // Done!
            if trow_begin >= trows { trow_begin=trow_min; }
            let mut trow=trow_begin;
    
            // let active_range = [trow_begin*tstride+tcol .. trow_begin*tstride+ttotal];
    
            /* Massage the active row so that it can eliminate every column */
            loop {
                /* Which single rows are available?
                 * All of them, unless we're on the first available tile-row
                 */
                let first_available_row =
                    if trow*Tile::EDGE_BITS < rank {
                        rank - trow*Tile::EDGE_BITS
                    } else { 0 };
                let rows_avail = ((1 as Edge)<<first_available_row).wrapping_neg();
    
                /* Make pointers to the current tile we're looking at to get more columns */
                let mut working = self.tiles[trow*tstride+tcol];
    
                /* If it's not the first, apply our progress so far */
                if trow != trow_begin {
                    self.rowop(working * &perm_matrix_cumulative, trow, trow_begin, tcol);
                    working = self.tiles[trow*tstride+tcol];
                }
    
                /* Row-reduce the current tile */
                let (aug, perm, ech_new) = working.pseudoinverse(rows_avail);
                debug_assert!(ech & ech_new == 0);

                if ech_new != 0 {
                    if trow == trow_begin {
                        /* We got some new columns.  Apply the operation to the rest of the row */
                        row_mul(aug, &mut self.tiles[trow*tstride+tcol .. trow*tstride+ttotal]);
                        perm_matrix_cumulative = perm;
                    } else {
                        /* Move the new progress to the active row */
                        let nech = ech.count_ones() as Index;
                        let nech_new = ech_new.count_ones() as Index;

                        /* Eliminate these columns from the active row, then re-add them as identity */
                        let factor = self.tiles[trow_begin*tstride+tcol] * &perm
                                   + Tile::IDENTITY.extract_cols(first_available_row,nech,nech_new);
                        self.rowop(factor * aug, trow_begin, trow, tcol);
    
                        /* Append the permutation matrix to perm */
                        (&mut perm_matrix_cumulative[nech..nech+nech_new])
                            .copy_from_slice(&perm[first_available_row .. first_available_row+nech_new]);
                    }
                    ech |= ech_new;
                }
    
                /* next row */
                trow += 1;
                if trow >= trows { trow=trow_min; } /* wrap around */
                if trow == trow_begin || ech == !0 { break; }
            }
    
            /* OK, we now have a tile which echelonizes all the selected columns.  Eliminate them. */
            for trow in 0..trows {
                if trow != trow_begin {
                    let factor = self.tiles[trow*tstride+tcol] * &perm_matrix_cumulative;
                    self.rowop(factor, trow, trow_begin, tcol);
                }
            }
    
            let begin = max(trow_begin * Tile::EDGE_BITS, rank);
            let active_tile = self.tiles[trow_begin*tstride+tcol];
            if !(active_tile & !Tile::IDENTITY).is_zero() {
                /* The working tile is permuted, because it picked up some rows from one tile-row,
                 * and other rows from another tile-row.  Unepermute it to make sure it's in REF */
                let rows_avail = ((1 as Edge)<<(begin % Tile::EDGE_BITS)).wrapping_neg();
                let (pseudoinv, _perm, _ech_new) = active_tile.pseudoinverse(rows_avail);
                row_mul(pseudoinv, &mut self.tiles[trow_begin*tstride+tcol .. trow_begin*tstride+ttotal]);
            }
    
            /* Swap the active row into place at the beginning of the matrix */
            let nech = ech.count_ones() as usize;
            self.swap_rows(rank, begin, nech, tcol);
            rank += nech;
    
            /* Mark the echelonized columns */
            for i in 0..Tile::EDGE_BITS {
                if ((ech>>i)&1) != 0 {
                    column_is_in_echelon.insert(tcol*Tile::EDGE_BITS + i);
                }
            }

        }
    
        (rank, column_is_in_echelon)
    }

    /* Place the columns of the other matrix next to this one's, and add their aug components */
    pub fn append_columns(&self, rhs: &Matrix) -> Matrix {
        assert!(self.rows == rhs.rows);
        let mut ret = Matrix::new(self.rows, self.cols_main + rhs.cols_main, max(self.cols_aug,rhs.cols_aug));
        let tcols_self = tiles_spanning(self.cols_main);
        let tcols_rhs  = tiles_spanning(rhs.cols_main);
        let tcols_ret  = tiles_spanning(ret.cols_main);
        let taugs_self = tiles_spanning(self.cols_aug);
        let taugs_rhs  = tiles_spanning(rhs.cols_aug);
        let trows      = tiles_spanning(self.rows);

        /* Construct permutations in case it's unaligned */
        let mut perm_to_cur  = PERMUTE_ALL_ZERO;
        let mut perm_to_prev = PERMUTE_ALL_ZERO;
        let bshift = self.cols_main % Tile::EDGE_BITS;
        if bshift != 0 {
            for i in 0..bshift {
                perm_to_cur[i] = (i + Tile::EDGE_BITS - bshift) as u8;
            }
            for i in 0..Tile::EDGE_BITS-bshift {
                perm_to_prev[i + bshift] = i as u8;
            }
        }

        for trow in 0..trows {
            /* Copy self to output */
            ret.tiles[ret.stride*trow .. ret.stride*trow + tcols_self].copy_from_slice(
                &self.tiles[self.stride*trow .. self.stride*trow+tcols_self]);

            /* Copy rhs to output */
            if self.cols_main % Tile::EDGE_BITS == 0 {
                /* Aligned: just memcpy it */
                ret.tiles[ret.stride*trow+tcols_self .. ret.stride*trow + tcols_self+tcols_rhs].copy_from_slice(
                    &rhs.tiles[rhs.stride*trow .. rhs.stride*trow+tcols_rhs]);
            } else {
                /* Unaligned: use column permutations */
                for tcol in 0..tcols_rhs {
                    ret.tiles[ret.stride*trow + tcols_self+tcol-1]  += rhs.tiles[rhs.stride*trow+tcol].permute_columns(&perm_to_prev);
                    if tcols_self+tcol < tcols_ret {
                        ret.tiles[ret.stride*trow + tcols_self+tcol] = rhs.tiles[rhs.stride*trow+tcol].permute_columns(&perm_to_cur);
                    }
                }
            }

            /* Combine the augcols */
            for taug in 0..taugs_self {
                ret.tiles[ret.stride*trow+tcols_ret+taug] = self.tiles[self.stride*trow+tcols_self+taug];
            }
            for taug in 0..taugs_rhs {
                ret.tiles[ret.stride*trow+tcols_ret+taug] += rhs.tiles[rhs.stride*trow+tcols_rhs+taug];
            }
        }
        ret
    }

    /**
     * Splits self into two matrices, nondestructively.
     * If want_yes, the first will contain columns where which_to_take == true.
     * If want_no, the second will contain columns where which_to_take == false, and the aug columns.
     * Unwanted outputs will be the trivial matrix.
     */
    pub fn partition_columns(
        &self, columns: &BitSet,
        columns_before: usize,
        columns_after: usize,
        want_yes: bool,
        want_no: bool
    ) -> (Matrix, Matrix) {
        let trows = tiles_spanning(self.rows);
        
        /* Allocate the results */
        let mut yes = if want_yes {
            Matrix::new(self.rows, columns.len(), 0)
        } else {
            Matrix::new(0,0,0)
        };

        let mut no = if want_no {
            let mut no = Matrix::new(self.rows,
                columns_before + columns_after + self.cols_main - columns.len(),
                self.cols_aug
            );
            /* Copy the aug cols to no */
            let aug_offset_result = tiles_spanning(no.cols_main);
            let aug_offset_self   = tiles_spanning(self.cols_main);
            for trow in 0..trows {
                for acol in 0..tiles_spanning(self.cols_aug) {
                    no.tiles[trow*no.stride+aug_offset_result+acol]
                        = self.tiles[trow*self.stride+aug_offset_self+acol];
                }
            }
            no
        } else {
            Matrix::new(0,0,0)
        };

        /* Count how many columns would be added before this one. */
        let offset_yes = columns.count_within(0..columns_before);
        let offset_no = columns_before - offset_yes;
        let mut util_yes = PartitionUtility::<Permutation>::new(want_yes, offset_yes, self.cols_main);
        let mut util_no  = PartitionUtility::<Permutation>::new(want_no,  offset_no,  self.cols_main);

        for col in 0..self.cols_main {
            /* Add the current column to the set */
            if columns.contains(col+columns_before) {
                util_yes.set(col);
            } else {
                util_no.set(col);
            }

            if let Some((perm,tcol,my_tcol)) = util_yes.apply(col) {
                for trow in 0..trows {
                    yes.tiles[trow*yes.stride + tcol]
                        |= self.tiles[trow*self.stride + my_tcol].permute_columns(&perm);
                }
            }

            if let Some((perm,tcol,my_tcol)) = util_no.apply(col) {
                for trow in 0..trows {
                    no.tiles[trow*no.stride + tcol]
                        |= self.tiles[trow*self.stride + my_tcol].permute_columns(&perm);
                }
            }
        }
        (yes,no)
    }

    /**
     * Splits self into two matrices, nondestructively.  The first will contain those rows
     * which are in the set, and the second those which are not.
     * 
     * Doesn't take offsets or want_* because libfrayed doesn't need them
     */
    pub fn partition_rows(&self, rows: &BitSet) -> (Matrix,Matrix) {
        let tcols = tiles_spanning(self.cols_main) + tiles_spanning(self.cols_aug);
        
        /* Allocate the results */
        let mut yes = Matrix::new(rows.len(),             self.cols_main, self.cols_aug);
        let mut no  = Matrix::new(self.rows - rows.len(), self.cols_main, self.cols_aug);

        let mut util_yes = PartitionUtility::<Tile>::new(true, 0, self.rows);
        let mut util_no  = PartitionUtility::<Tile>::new(true, 0, self.rows);

        for row in 0..self.rows {
            /* Add the current column to the set */
            if rows.contains(row) {
                util_yes.set(row);
            } else {
                util_no.set(row);
            }

            if let Some((perm,trow,my_trow)) = util_yes.apply(row) {
                row_mul_acc(&mut yes.tiles[trow*yes.stride .. trow*yes.stride+tcols], perm,
                            &self.tiles[my_trow*self.stride .. my_trow*self.stride+tcols]);
            }

            if let Some((perm,trow,my_trow)) = util_no.apply(row) {
                row_mul_acc(&mut no.tiles[trow*no.stride .. trow*no.stride+tcols], perm,
                            &self.tiles[my_trow*self.stride .. my_trow*self.stride+tcols]);
            }
        }
        (yes,no)
    }

    /** Return two matrices: one with the rows of self less than row, and one with more than row */
    pub fn split_at_row(&self, row:usize) -> (Matrix, Matrix) {
        debug_assert!(row <= self.rows);
        let tcols = tiles_spanning(self.cols_main) + tiles_spanning(self.cols_aug);
        let mut top = Matrix::new(row,           self.cols_main, self.cols_aug);
        let mut bot = Matrix::new(self.rows-row, self.cols_main, self.cols_aug);
        for trow in 0..tiles_spanning(row) {
            top.tiles[trow*top.stride .. trow*top.stride+tcols]
                .copy_from_slice(& self.tiles[trow*self.stride .. trow*self.stride+tcols]);
        }

        if row % Tile::EDGE_BITS != 0 {
            /* Mask the top */
            let row0 = row % Tile::EDGE_BITS;
            let mask = Tile::rows_mask(0,row0);
            let trow = row/Tile::EDGE_BITS;
            for tcol in 0..tcols {
                top.tiles[trow*top.stride + tcol] &= mask;
            }

            /* Shift the bottom */
            let perm_up = Tile::IDENTITY.extract_cols(0, Tile::EDGE_BITS-row0, row0);
            let perm_down = Tile::IDENTITY.extract_cols(row0, 0, Tile::EDGE_BITS-row0);
            let off = tiles_spanning(row);
            let tot_rows = tiles_spanning(self.rows);
            for trow in 0..tiles_spanning(bot.rows) {
                let mut bot_row = &mut bot.tiles[trow*bot.stride .. trow*bot.stride+tcols];
                row_mul_acc(&mut bot_row, perm_down, &self.tiles[(trow+off-1)*self.stride .. (trow+off-1)*self.stride+tcols]);
                if trow+off < tot_rows  {
                    row_mul_acc(&mut bot_row, perm_up,   &self.tiles[(trow+off)*self.stride .. (trow+off)*self.stride+tcols]);
                }
            }
        } else {
            /* Memcpy the bottom */
            let off = tiles_spanning(row);
            for trow in 0..tiles_spanning(bot.rows) {
                bot.tiles[trow*bot.stride .. trow*bot.stride+tcols]
                    .copy_from_slice(& self.tiles[(trow+off)*self.stride .. (trow+off)*self.stride+tcols]);
            }
        }

        (top,bot)
    }

    /**
     * The opposite of partition_rows.
     * Interleave self's rows and other's rows to form a new matrix;
     * take the rows from self according to the bitset, or from other.
     */
    pub fn interleave_rows(&self, other: &Matrix, take_from_self: &BitSet) -> Matrix {
        debug_assert_eq!(take_from_self.len(), self.rows);
        debug_assert_eq!(self.cols_main, other.cols_main);
        debug_assert_eq!(self.cols_aug, other.cols_aug);

        let tcols = tiles_spanning(self.cols_main) + tiles_spanning(self.cols_aug);
        
        /* Allocate the results */
        let mut result = Matrix::new(self.rows + other.rows, self.cols_main, self.cols_aug);

        let mut util_yes = PartitionUtility::<Tile>::new(true, 0, result.rows);
        let mut util_no  = PartitionUtility::<Tile>::new(true, 0, result.rows);

        for row in 0..result.rows {
            /* Add the current column to the set */
            if take_from_self.contains(row) {
                util_yes.coset(row);
            } else {
                util_no.coset(row);
            }

            if let Some((perm,trow,my_trow)) = util_yes.apply(row) {
                row_mul_acc(&mut result.tiles[my_trow*result.stride .. my_trow*result.stride+tcols], perm,
                            &self.tiles[trow*self.stride .. trow*self.stride+tcols]);
            }

            if let Some((perm,trow,my_trow)) = util_no.apply(row) {
                row_mul_acc(&mut result.tiles[my_trow*result.stride .. my_trow*result.stride+tcols], perm,
                            &other.tiles[trow*other.stride .. trow*other.stride+tcols]);
            }
        }
        result
    }

    /**
     * Puts self in row-reduced echelon form.
     * Then if self has full row-rank, then return its systematic form.
     * Otherwise return None.
     */
    pub fn systematic_form(&mut self) -> Option<Systematic> {
        let (rank, column_is_in_echelon) = self.rref();
        if rank < self.rows {
            None
        } else {
            Some (Systematic {
                rhs: self.partition_columns(&column_is_in_echelon, 0, 0, false, true).1,
                echelon: column_is_in_echelon
            })
        }
    }
}

/** Implement += */
impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other:&Matrix) { self.add_assign(other); }
}


/**************************************************************************
 * Partitioning utility
 **************************************************************************/
/** Some way to permute a tile */
trait PermLike {
    const EMPTY : Self;
    fn set(&mut self, from: usize, to:usize);
}

impl PermLike for Tile {
    const EMPTY : Tile = Tile::ZERO;
    #[inline(always)]
    fn set(&mut self, from:usize, to:usize) { self.set_bit(from,to); }
}

impl PermLike for Permutation {
    const EMPTY : Permutation = PERMUTE_ALL_ZERO;
    #[inline(always)]
    fn set(&mut self, from:usize, to:usize) { self[from] = to as u8; }
}

/**
 * Structure for tracking the current state of partitioning / interleaving
 * rows/columns of a matrix.
 */
struct PartitionUtility<P>  {
    want : bool, /* are we even using this? */
    nonempty : bool,
    current : P,
    position : usize,
    total_positions : usize
}

impl <P: PermLike+Copy> PartitionUtility<P> {
    /* Create a new PartitionUtility starting from the given position */
    #[inline(always)]
    fn new(want:bool, start_position:usize, total_positions:usize) -> Self {
        Self { want:want, nonempty:false, current : P::EMPTY,
            position: start_position,
            total_positions:total_positions
        }
    }

    /* Mark a row/column to be moved */
    #[inline(always)]
    fn set(&mut self, ipt: usize) {
        self.current.set(self.position % Tile::EDGE_BITS, ipt % Tile::EDGE_BITS);
        self.nonempty = self.want;
        self.position += 1;
    }

    /* As set, but the permutation goes the other way */
    #[inline(always)]
    fn coset(&mut self, ipt: usize) {
        self.current.set(ipt % Tile::EDGE_BITS, self.position % Tile::EDGE_BITS);
        self.position += 1;
        self.nonempty = self.want;
    }

    /* Do we need to apply it now?  If so, return what to apply, and to which tile */
    #[inline(always)]
    fn apply(&mut self, position: usize) -> Option<(P,usize,usize)> {
        if !self.nonempty
            ||     ((position+1) != self.total_positions
                 && (position+1)  % Tile::EDGE_BITS != 0
                 && self.position % Tile::EDGE_BITS != 0) {
            return None;
        }
        self.nonempty = false;
        let to_apply = self.current;
        self.current = P::EMPTY;
        Some((to_apply,(self.position-1)/Tile::EDGE_BITS,position/Tile::EDGE_BITS))
    }
}

/**************************************************************************
 * Matrices in systematic form
 **************************************************************************/

/** A matrix in systematic form, with the systematic columns omitted. */
pub struct Systematic {
    pub rhs     : Matrix,
    pub echelon : BitSet
}

impl Systematic {
    /** Return the (trivial) systematic identity matrix of a given size */
    pub fn identity(size:usize) -> Systematic {
        Systematic {
            rhs: Matrix::new(size,0,0),
            echelon: BitSet::from_range(size,0..size)
        }
    }

    /** Total columns, including the ones in echelon */
    pub fn total_cols_main(&self) -> usize {
        self.rhs.cols_main + self.rhs.rows
    }

    /** Project out `self` from `group` by eliminating the columns of sys that are in echelon.
     * Group should be at most as large as sys; align it to the left or right as indicated.
     */
    pub fn project_out(&self, group: &Matrix, is_left: bool) -> Matrix {
        let sys_total_cols = self.total_cols_main();
        assert!(group.cols_main <= sys_total_cols);
        let (before, after) = if is_left {
            (0, sys_total_cols - group.cols_main)
        } else {
            (sys_total_cols - group.cols_main, 0)
        };

        /* The ones in echelon are canceled out by applying sys.rhs */
        let (to_multiply, mut to_copy) = group.partition_columns(&self.echelon, before, after, true, true);
        to_copy.accum_mul(&to_multiply, &self.rhs);
        to_copy
    }
}

/**************************************************************************
 * Tests
 **************************************************************************/

#[cfg(test)]
mod tests {
    use crate::tilematrix::matrix::Matrix;
    use rand::{Rng,thread_rng};
    use crate::tilematrix::bitset::BitSet;

    /** Generate a random bitset */
    fn random_subset(n:usize) -> BitSet {
        let mut ret = BitSet::with_capacity(n);
        for i in 0..n {
            if thread_rng().gen::<bool>() {
                ret.insert(i);
            }
        }
        ret
    }

    /** Test that matrix multiplication is linear */
    #[test]
    fn test_matmul() {
        for a in 1..=8usize {
            for b in 1..=8usize {
                for c in 1..=8usize {
                    let aug = (a+b+c)%5;
                    let mut x0 = Matrix::new(a*8,b*8,aug*8);
                    let mut x1 = Matrix::new(a*8,b*8,aug*8);
                    let mut x2 = Matrix::new(a*8,b*8,aug*8);
                    let mut y0 = Matrix::new(b*8,c*8,aug*8);
                    let mut y1 = Matrix::new(b*8,c*8,aug*8);
                    let mut y2 = Matrix::new(b*8,c*8,aug*8);
                    x0.randomize();
                    x1.randomize();
                    x2.randomize();
                    y0.randomize();
                    y1.randomize();
                    y2.randomize();

                    let mut indiv = x0.mul(&y0);
                    indiv += &x0.mul(&y1);
                    indiv += &x0.mul(&y2);
                    y0 += &y1;
                    y0 += &y2;
                    let combined = x0.mul(&y0);
                    assert_eq!(indiv,combined);

                    let mut indiv = x0.mul(&y0);
                    indiv += &x1.mul(&y0);
                    indiv += &x2.mul(&y0);
                    x0 += &x1;
                    x0 += &x2;
                    let combined = x0.mul(&y0);
                    assert_eq!(indiv,combined);
                }
            }
        }
    }

    /** Tests that row-reducing makes sense */
    #[test]
    fn test_rref() {
        for a in 1..=10usize {
            for b in 1..=10usize {
                let mut x0 = Matrix::new(a*8,b*8,b*8);
                x0.randomize();
                let (rank,in_ech) = x0.rref();

                assert_eq!(rank, in_ech.len());
                assert!(rank <= x0.rows);
                assert!(rank <= x0.cols_main);

                /* Check that it's really in echelon form */
                let mut which_row = 0;
                for ech in in_ech.iter() {
                    for row in 0..x0.rows {
                        assert_eq!(x0.get_bit(row,ech), row==which_row);
                    }
                    which_row += 1;
                }

                /* TODO: somehow check that the echelon form matches??? */
            }
        }
    }

    /** Just a sanity check; should inherit most properties from rref */
    #[test]
    fn test_systematic() {
        for a in 1..=10usize {
            for b in 1..=10usize {
                let mut x0   = Matrix::new(a*8,b*8,b*8);
                let proj = Matrix::new(a*8,b*8,b*8);
                x0.randomize();
                let sys = x0.systematic_form();
                match sys {
                    None => {},
                    Some(sys) => {
                        assert_eq!(sys.rhs.rows, x0.rows);
                        assert_eq!(sys.rhs.cols_aug, x0.cols_aug);
                        assert_eq!(sys.rhs.cols_main, x0.cols_main - x0.rows);
                        assert_eq!(sys.echelon.len(), x0.rows);
                        sys.project_out(&proj, true);
                    }
                }
            }
        }
    }

    /** Test partitioning matrices */
    #[test]
    fn test_partition() {
        for a in 1..=10usize {
            for b in 1..=10usize {
                /* Test partition columns */
                let m = Matrix::new(a*15,b*15,a*15);
                let s = random_subset(b*15);
                let t = random_subset(a*15);
                let (yes,no) = m.partition_columns(&s,0,0,true,true);
                let (mut cyes, mut cno) = (0,0);
                assert_eq!(yes.cols_aug, 0);
                assert_eq!(yes.cols_main + no.cols_main, m.cols_main);
                assert_eq!(no.cols_aug, m.cols_aug);

                for col in 0..m.cols_main {
                    let (yn, cyn) = if s.contains(col) {
                        cyes += 1;
                        (&yes, cyes-1)
                    } else {
                        cno += 1;
                        (&no, cno-1)
                    };
                    for row in 0..m.rows {
                        assert_eq!(m.get_bit(row,col),yn.get_bit(row,cyn));
                    }
                }

                for aug in 0..m.cols_aug {
                    for row in 0..m.rows {
                        assert_eq!(m.get_aug_bit(row,aug),no.get_bit(row,aug));
                    }
                }

                /* Test partition rows */
                let (yes,no) = m.partition_rows(&t);
                let (mut ryes, mut rno) = (0,0);
                assert_eq!(yes.cols_aug,  m.cols_aug);
                assert_eq!(no.cols_aug,   m.cols_aug);
                assert_eq!(yes.cols_main, m.cols_main);
                assert_eq!(no.cols_main,  m.cols_main);
                assert_eq!(yes.rows + no.rows, m.rows);
                for row in 0..m.rows {
                    let (yn, ryn) = if t.contains(row) {
                        ryes += 1;
                        (&yes, ryes-1)
                    } else {
                        rno += 1;
                        (&no, rno-1)
                    };
                    for col in 0..m.cols_main {
                        assert_eq!(m.get_bit(row,col),yn.get_bit(ryn,col));
                    }
                    for aug in 0..m.cols_aug {
                        assert_eq!(m.get_aug_bit(row,aug),yn.get_aug_bit(ryn,aug));
                    }
                }

                assert_eq!(m, yes.interleave_rows(&no, &t));
            }
        }
    }

    /** Test that appending columns works */
    #[test]
    fn test_append() {
        for a in 1..=10usize {
            for b in 1..=10usize {
                for c in 1..=10usize {
                    let l = Matrix::new(a*8,b*8,a*8);
                    let r = Matrix::new(a*8,c*8,a*8);
                    let a = l.append_columns(&r);

                    assert_eq!(a.rows,l.rows);
                    assert_eq!(a.cols_main, l.cols_main+r.cols_main);
                    assert_eq!(a.cols_aug, l.cols_aug);

                    for row in 0..a.rows {
                        for col in 0..l.cols_main {
                            assert_eq!(a.get_bit(row,col), l.get_bit(row,col));
                        }
                        for col in 0..r.cols_main {
                            assert_eq!(a.get_bit(row,col+l.cols_main), r.get_bit(row,col));
                        }
                        for aug in 0..a.cols_aug {
                            assert_eq!(a.get_aug_bit(row,aug),
                                r.get_aug_bit(row,aug)^l.get_aug_bit(row,aug));
                        }
                    }
                }
            }
        }
    }
}