/*
 * @file tile.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Single-tile matrix operations.  This file implements operations on 16x16 matrices,
 * represented by four 64-bit integers.  The bits within a subtile are in column-major
 * order, and the tiles within a row are in row-major order (sorry, that's just how
 * it worked out)
 * That is, column 0 is bits 0..7 of the tile 0 and of tile 2.
 */

/**************************************************************************
 * Single-tile operations
 **************************************************************************/
use std::ops::{Add,AddAssign,Mul,BitAnd,BitAndAssign,BitXor,BitXorAssign,Not,BitOr,BitOrAssign};
use rand::Rng;
use rand::distributions::{Distribution,Standard};
use std::cmp::{min};
 
pub type Edge  = u16; // Occasionally actually stored
pub type Index = usize;
type Storage = u64;
const LINEAR_DIM  : usize = 2;
const SUBTILE_DIM : usize = 8;
const STORAGE_PER : usize = LINEAR_DIM*LINEAR_DIM;
const COL_MASK    : Storage = 0xFF;
const ROW_MASK    : Storage = 0x0101010101010101;

pub type Permutation = [u8; LINEAR_DIM*SUBTILE_DIM];
pub const PERMUTE_ZERO : u8 = 0xFF;
pub const PERMUTE_ALL_ZERO : Permutation = [PERMUTE_ZERO; LINEAR_DIM*SUBTILE_DIM];

/** 16x16 GF(2) matrix */
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Tile {
    pub storage : [Storage; STORAGE_PER] // 2x2 column-major
}

impl Tile {
    pub const ZERO : Tile = Tile { storage: [0;STORAGE_PER] };
    pub const FULL : Tile = Tile { storage: [!0u64; STORAGE_PER] };
    pub const IDENTITY : Tile = Tile { storage: [0x8040201008040201,0,0,0x8040201008040201] };
    pub const EDGE_BITS: Index = Edge::BITS as Index;

    /** Set one bit of the tile */
    #[inline(always)]
    pub fn set_bit(&mut self, row:Index,col:Index) {
        self.storage[(row/SUBTILE_DIM) * LINEAR_DIM + col/SUBTILE_DIM]
            |= 1 << ((col%SUBTILE_DIM)*8+(row%SUBTILE_DIM));
    }

    /** Clear one bit of the tile */
    #[inline(always)]
    pub fn clear_bit(&mut self, row:Index,col:Index) {
        self.storage[(row/SUBTILE_DIM) * LINEAR_DIM + col/SUBTILE_DIM]
            &= !(1 << ((col%SUBTILE_DIM)*8+(row%SUBTILE_DIM)));
    }

    /**Toggle one bit of the tile */
    #[inline(always)]
    pub fn toggle_bit(&mut self, row:Index,col:Index) {
        self.storage[(row/SUBTILE_DIM) * LINEAR_DIM + col/SUBTILE_DIM]
            ^= 1 << ((col%SUBTILE_DIM)*8+(row%SUBTILE_DIM));
    }

    /** Return a tile with a single bit set */
    pub fn single_bit(row:Index,col:Index) -> Tile {
        let mut ret = Tile::ZERO; ret.set_bit(row,col); ret
    }

    /** Return the transpose of a storage sub tile */
    fn transpose1(a:Storage) -> Storage {
        let a = (a & 0xF0F0F0F00F0F0F0F)
        | ((a << 28) & 0x0F0F0F0F00000000)
        | ((a & 0x0F0F0F0F00000000)>>28); // transpose blocks of size 4

        let a = (a & 0xCCCC3333CCCC3333)
        | ((a << 14) & 0x3333000033330000)
        | ((a & 0x3333000033330000)>>14); // size 2

        let a = (a & 0xAA55AA55AA55AA55)
        | ((a << 7) & 0x5500550055005500)
        | ((a & 0x5500550055005500)>>7); // size 1

        a
    }

    /** Return the transpose of self */
    pub fn transpose(self) -> Tile {
        let mut ret = Tile::ZERO;
        for row in 0..LINEAR_DIM {
            for col in 0..LINEAR_DIM {
                ret.storage[row*LINEAR_DIM+col] = Tile::transpose1(self.storage[col*LINEAR_DIM+row]);
            }
        }
        ret
    }

    /** Permute our columns in place.  Slow due to vectorization checks */
    fn mut_permute_columns(&mut self, perm: &Permutation) {
        tile_mut_permute_columns::runtime(self, perm);
    }

    /** Permute our columns out of place */
    pub fn permute_columns(self, perm: &Permutation) -> Self {
        let mut ret = self;
        ret.mut_permute_columns(perm);
        ret
    }

    fn broadcast_row1(storage:Storage, row:Index) -> Storage {
        let a = (storage>>row) & ROW_MASK;
        (a << SUBTILE_DIM).wrapping_sub(a)
    }

    /** Return a tile where all rows are copies of the selected one */
    fn broadcast_row(self, row:Index) -> Self {
        let mut ret = Tile::ZERO;
        let row1 = row / SUBTILE_DIM;
        let row0 = row % SUBTILE_DIM;
        for row in 0..LINEAR_DIM {
            for col in 0..LINEAR_DIM {
                ret.storage[row*LINEAR_DIM+col] = Tile::broadcast_row1(self.storage[row1*LINEAR_DIM+col], row0);
            }
        }
        ret
    }

    /** Return a tile where all columns are copies of the selected one */
    fn broadcast_col1(storage:Storage, col:Index) -> Storage {
        ((storage>>(8*col)) & 0xFF).wrapping_mul(ROW_MASK)
    }

    /** Return a tile where all columns are copies of the selected one */
    pub fn broadcast_col(self, col:Index) -> Self {
        let mut ret = Tile::ZERO;
        let col1 = col / SUBTILE_DIM;
        let col0 = col % SUBTILE_DIM;
        for row in 0..LINEAR_DIM {
            for col in 0..LINEAR_DIM {
                ret.storage[row*LINEAR_DIM+col] = Tile::broadcast_col1(self.storage[row*LINEAR_DIM+col1], col0);
            }
        }
        ret
    }

    /** Return one bit of the tile as a u64 */
    pub fn get_bit(self, row:Index, col:Index) -> bool {
        (1 & (self.storage[(row/SUBTILE_DIM)*LINEAR_DIM + col/SUBTILE_DIM]
            >> ((col%SUBTILE_DIM)*8+(row%SUBTILE_DIM)))) != 0
    }

    /** Return a tile where all columns are copies of the given Edge */
    pub fn broadcast_edge_as_col(edge:Edge) -> Tile {
        let mut ret = Tile::ZERO;
        for row in 0..LINEAR_DIM {
            let edge_promoted = (edge as Storage)>>(row*SUBTILE_DIM) & COL_MASK;
            let edge_promoted = edge_promoted * ROW_MASK;
            for col in 0..LINEAR_DIM {
                ret.storage[row*LINEAR_DIM+col] = edge_promoted;
            }
        }
        ret
    }

    /** Return a tile where all rows are copies of the given Edge */
    pub fn broadcast_edge_as_row(edge:Edge) -> Tile {
        let mut ret = Tile::ZERO;
        for col in 0..LINEAR_DIM {
            let edge_promoted = (edge as Storage)>>(col*SUBTILE_DIM) & COL_MASK;
            let edge_promoted = (edge_promoted & 0xFE).wrapping_mul(0x2040810204081) | edge_promoted;
            let edge_promoted = (edge_promoted & ROW_MASK) * COL_MASK;
            for row in 0..LINEAR_DIM {
                ret.storage[row*LINEAR_DIM+col] = edge_promoted;
            }
        }
        ret
    }

    /** Set a row of self to a given edge */
    pub fn mut_set_row(&mut self, edge:Edge, row:Index) {
        let row1 = row / SUBTILE_DIM;
        let row0 = row % SUBTILE_DIM;
        let submask = ROW_MASK << row0;
        for col in 0..LINEAR_DIM {
            let edge_promoted = (edge as Storage)>>(col*SUBTILE_DIM) & COL_MASK;
            let edge_promoted = (edge_promoted & 0xFE).wrapping_mul(0x2040810204081) | edge_promoted;
            let edge_promoted = (edge_promoted & ROW_MASK) << row0;
            self.storage[row1*LINEAR_DIM+col] = (self.storage[row1*LINEAR_DIM+col] & !submask) | edge_promoted;
        }
    }

    /** Return a tile which is one on the given row and zero elsewhere */
    pub fn row_mask(row:Index) -> Tile {
        let mut ret = Tile::ZERO;
        let row1 = row / SUBTILE_DIM;
        let submask = ROW_MASK << (row % SUBTILE_DIM);
        for col in 0..LINEAR_DIM {
            ret.storage[row1*LINEAR_DIM+col] = submask;
        }
        ret
    }

    /** Return a tile which is one on the given column and zero elsewhere */
    pub fn col_mask(col:Index) -> Tile {
        let mut ret = Tile::ZERO;
        let col1 = col / SUBTILE_DIM;
        let submask = 0xFF << (8*(col % SUBTILE_DIM));
        for row in 0..LINEAR_DIM {
            ret.storage[row*LINEAR_DIM+col1] = submask;
        }
        ret
    }

    /** Return a tile which is one on [row, row+n-1] and zero elsewhere */
    pub fn rows_mask(row:Index, n:Index) -> Tile {
        Tile::broadcast_edge_as_col(((1u64<<(row+n))-(1<<row)) as Edge)
    }

    /** Return a tile which is one on [col, col+n-1] and zero elsewhere */
    pub fn cols_mask(col:Index, n:Index) -> Tile {
        Tile::broadcast_edge_as_row(((1u64<<(col+n))-(1<<col)) as Edge)
    }
  
    /** Swap row a and row b */
    pub fn mut_swap_row(&mut self, rowa:Index, rowb:Index) {
        let (rowa0,rowa1) = (rowa % SUBTILE_DIM, rowa / SUBTILE_DIM);
        let (rowb0,rowb1) = (rowb % SUBTILE_DIM, rowb / SUBTILE_DIM);
        let ma = ROW_MASK << rowa0;
        let mb = ROW_MASK << rowb0;
        if rowa1 == rowb1 {
            for col in 0..LINEAR_DIM {
                let sub = self.storage[rowa1*LINEAR_DIM+col];
                let sub = (sub & !ma & !mb)
                        | (mb & ((sub >> rowa0) << rowb0))
                        | (ma & ((sub >> rowb0) << rowa0));
                self.storage[rowa1*LINEAR_DIM+col] = sub;
            }
        } else {
            for col in 0..LINEAR_DIM {
                let suba = self.storage[rowa1*LINEAR_DIM+col];
                let subb = self.storage[rowb1*LINEAR_DIM+col];
                self.storage[rowa1*LINEAR_DIM+col] = (suba & !ma) | (((subb >> rowb0) << rowa0) & ma);
                self.storage[rowb1*LINEAR_DIM+col] = (subb & !mb) | (((suba >> rowa0) << rowb0) & mb);
            }
        }
    }

    /** Extract self.cols[colb..colb+ncols] as cola..cola+ncols */
    pub fn extract_cols(&self, cola:Index, colb:Index, ncols:usize) -> Tile {
        /* PERF: vectorize? */
        let mut permo = PERMUTE_ALL_ZERO;
        for i in 0..Tile::EDGE_BITS {
            if i >= cola && i < cola+ncols {
                permo[i] = (i-cola+colb) as u8;
            }
        }
        *self * &permo
    }

    /** Self ^= colb of b moved to cola.  Used to implement permutations. */
    #[allow(dead_code)]
    pub fn mut_xor_col(&mut self, b: &Tile, cola:Index, colb:Index) {
        let (cola0,cola1) = (cola%SUBTILE_DIM, cola/SUBTILE_DIM);
        let (colb0,colb1) = (colb%SUBTILE_DIM, colb/SUBTILE_DIM);
        for row in 0..LINEAR_DIM {
            self.storage[row*LINEAR_DIM+cola1] ^=
                (b.storage[row*LINEAR_DIM+colb1]
                    >> (colb0*SUBTILE_DIM)
                    & COL_MASK)
                    << (cola0*SUBTILE_DIM);
        }
    }

    /** Get the index of the first nonzero entry in a column. */
    pub fn first_nonzero_entry_in_col(self, col:Index, mask:Edge) -> Option<Index> {
        let (col0,col1) = (col%SUBTILE_DIM, col/SUBTILE_DIM);
        for row in 0..LINEAR_DIM {
            let t = self.storage[row*LINEAR_DIM+col1] >> (SUBTILE_DIM * col0)
                  & COL_MASK & (mask>>SUBTILE_DIM*row) as Storage;
            if t != 0 {
                return Some((t.trailing_zeros() as Index) + row*SUBTILE_DIM);
            }
        }
        None
    }

    /** Return true if value is zero */
    pub fn is_zero(self) -> bool { self.storage == [0; STORAGE_PER] }

    /**
     * Compute pseudoinverse of a tile.
     * This function takes extra parameters
     * so that it can be used in an echelon form solver.
     * 
     * Ignore rows that are 0 in the rows_avail mask.
     * Return (pseudoinverse, permutation, echelon mask).
     * The echelon mask shows which columns the pseuoinverse puts into echelon.
     * The function will put as many columns into echelon as possible
     * The returned values satisfy: pseudoinverse * self * permutation is diagonal
     * perm is a permutation, and is a subset of brodcast_edge_as_col(ech)
     *    and of broadcast_edge_as_row(rows_avail)
     * 
     * TODO PERF: optimize now that we have subtiles
     */
    pub fn pseudoinverse (self, rows_avail: Edge) -> (Tile, Permutation, Edge) {
        let mut tile = self;
        let mut rows_avail = rows_avail;
        let mut aug = Tile::IDENTITY;
        let mut perm = PERMUTE_ALL_ZERO;
        let mut ech : Edge = 0;

        for col in 0 .. Tile::EDGE_BITS {
            /* Eliminate column by column. */
            let row_found = match tile.first_nonzero_entry_in_col(col,rows_avail) {
                None => continue,
                Some(r) => r
            };

            ech |= 1<<col;
            let row = rows_avail.trailing_zeros() as Index;
            /* PERF: avoid this and permute the rows at the end? */
            if row_found != row {
                tile.mut_swap_row(row,row_found);
                aug. mut_swap_row(row,row_found);
            }

            rows_avail ^= 1<<row; // &~, but it's currently set so ^= is the same

            /* Clear the tile and augmented tile */
            let mut tile2 = tile;
            tile2.toggle_bit(row,col);
            perm[row] = col as u8;
            let rows_affected = Tile::broadcast_col(tile2,col);
            tile ^= Tile::broadcast_row(tile,row) & rows_affected;
            aug  ^= Tile::broadcast_row(aug, row) & rows_affected;
        }
        (aug,perm,ech)
    }

    /** Print self for debug */
    #[allow(dead_code)]
    pub fn print(&self, name:&str) {
        print!("Matrix {}:\n", name);
        for i in 0..Tile::EDGE_BITS {
            for j in 0..Tile::EDGE_BITS {
                print!("{}", self.get_bit(i,j) as i8);
            }
            print!("\n");
        }
        print!("\n");
    }
}

/* GF2 matrix mul */
impl Mul for Tile {
    type Output = Self;
    fn mul(self, rhs:Self) -> Self {
        let mut ret : Tile = Tile::ZERO;
        for i in 0..Tile::EDGE_BITS {
            ret += self.broadcast_col(i) & rhs.broadcast_row(i);
        }
        ret
    }
}

impl Mul<&Permutation> for Tile {
    type Output = Self;
    fn mul(self, rhs:&Permutation) -> Self { self.permute_columns(rhs) }
}

impl BitXorAssign for Tile { fn bitxor_assign(&mut self, rhs:Self) {
    for i in 0..STORAGE_PER { self.storage[i] ^= rhs.storage[i]; }
}}
impl AddAssign    for Tile { fn add_assign   (&mut self, rhs:Self) {
    for i in 0..STORAGE_PER { self.storage[i] ^= rhs.storage[i]; }
}}
impl BitAndAssign for Tile { fn bitand_assign(&mut self, rhs:Self) {
    for i in 0..STORAGE_PER { self.storage[i] &= rhs.storage[i]; }
}}
impl BitOrAssign  for Tile { fn bitor_assign (&mut self, rhs:Self) {
    for i in 0..STORAGE_PER { self.storage[i] |= rhs.storage[i]; }
}}

impl BitXor for Tile {
    type Output = Self;
    fn bitxor(self, rhs:Self) -> Self { let mut ret = self; ret ^= rhs; ret }
}

/* Add is the same as xor */
impl Add for Tile {
    type Output = Self;
    fn add(self, rhs:Self) -> Self { let mut ret = self; ret += rhs; ret }
}

impl BitAnd for Tile {
    type Output = Self;
    fn bitand(self, rhs:Self) -> Self { let mut ret = self; ret &= rhs; ret }
}

impl BitOr for Tile {
    type Output = Self;
    fn bitor(self, rhs:Self) -> Self { let mut ret = self; ret |= rhs; ret }
}

impl Not for Tile {
    type Output = Self;
    fn not(self) -> Self { let mut ret = self; ret += Tile::FULL; ret }
}

impl Distribution<Tile> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Tile {
        let mut ret = Tile::ZERO;
        for i in 0..STORAGE_PER { ret.storage[i] = rng.gen::<Storage>(); }
        ret
    }
}

/****************************************************************************
 * Accelerators
 ****************************************************************************/

/** Methods supplied by vector accelerators */
pub trait Accelerator where {
    type MulTable: Clone+Copy;
    /// Determine whether this accelerator is available, e.g. whether the
    /// current machine has AVX2 or NEON
    fn is_available() -> bool { false }
    /// Permute the columns of a tile according to a permutation.
    /// Unsafe because: may use intrinsics
    unsafe fn mut_permute_columns(t:&mut Tile, permutation:&Permutation) {
        Scalar::mut_permute_columns(t,permutation)
    }
    /// Precompute a table for accelerating multiplication by t.
    /// Unsafe because: may use intrinsics
    unsafe fn compile_mul_table(_t:Tile) -> Self::MulTable {
        unimplemented!()
    }
    /// Perform precomputed multiplication by the table used to construct t.
    /// Unsafe because: may use intrinsics
    unsafe fn precomputed_mul(_table: &Self::MulTable, _tv: Tile) -> Tile {
        unimplemented!()
    }
    /// Permute the columns of a tile according to a permutation, without mutating it.
    /// Unsafe because: may use intrinsics
    unsafe fn permute_columns(mut t:Tile, permutation:&Permutation) -> Tile {
        Self::mut_permute_columns(&mut t,permutation);
        t
    }

}

/** SIMD accelerator macro, based on simdeez simd_runtime_generate macro */
#[macro_use]
pub mod simd {
    macro_rules! accelerated {
    ($vis:vis fn $fn_name:ident ($($arg:ident:$typ:ty),* $(,)? ) $(-> $rt:ty)? $body:block  ) => {
    $vis mod $fn_name {
        #[cfg(any(target_arch="aarch64",target_arch="armv7"))]
        use crate::tilematrix::tile::neon;
        #[cfg(any(target_arch="x86_64",target_arch="x86"))]
        use crate::tilematrix::tile::avx2;
        use crate::tilematrix::tile::{Accelerator,Scalar};
        use super::*;

        /// Run a function using intrinsics beloning to a particular acceleration package.
        /// Unsafe because: may use intrinsics
        #[allow(dead_code)]
        #[inline(always)]
        pub unsafe fn with_accel<Accel:Accelerator>($($arg:$typ,)*) $(-> $rt)? $body

        #[allow(dead_code)]
        pub fn with_scalar($($arg:$typ,)*) $(-> $rt)? {
            // Safe because: scalar code isn't really unsafe, so encapsulate here
            unsafe { with_accel::<Scalar>($($arg,)*) }
        }
        
        /// Run a function using AVX2 intrinsics
        /// Unsafe because: uses AVX2 intrinsics; will catch SIGILL if AVX2 is unspported.
        #[allow(dead_code)]
        #[cfg(any(target_arch="x86_64",target_arch="x86"))]
        #[target_feature(enable = "avx2")]
        pub unsafe fn with_avx2($($arg:$typ,)*) $(-> $rt)? {
            with_accel::<avx2::Avx2>($($arg,)*)
        }

        /// Run a function using AArch64 NEON intrinsics
        /// Safe (probably) because as far as I know, all AArch64 machines have NEON?
        #[allow(dead_code)]
        #[cfg(any(target_arch="aarch64"))]
        // it always has neon?
        // #[target_feature(enable = "neon")]
        pub fn with_neon($($arg:$typ,)*) $(-> $rt)? {
            unsafe { with_accel::<neon::Neon>($($arg,)*) }
        }

        /// Run a function using NEON intrinsics
        /// Unsafe because: uses NEON intrinsics; will catch SIGILL if NEON is unspported.
        #[allow(dead_code)]
        #[cfg(any(target_arch="armv7"))]
        #[target_feature(enable = "neon")]
        pub unsafe fn with_neon($($arg:$typ,)*) $(-> $rt)? {
            with_accel::<neon::Neon>($($arg,)*)
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub fn runtime($($arg:$typ,)*) $(-> $rt)? {
            #[cfg(target_arch="armv7")]
            if neon::Neon::is_available() {
                // Safe because: we checked that NEON is available
                unsafe { return with_neon($($arg,)*); }
            }
            #[cfg(target_arch="aarch64")]
            if neon::Neon::is_available() {
                // Safe because: NEON is always available (?)
                return with_neon($($arg,)*);
            }
            #[cfg(any(target_arch="x86_64",target_arch="x86"))]
            if avx2::Avx2::is_available() {
                // Safe because: we checked that AVX2 is available
                unsafe { return with_avx2($($arg,)*); }
            }
            with_scalar($($arg,)*)
        }
    }}}

    #[warn(unused_imports)]
    pub(crate) use accelerated;
}

accelerated!(pub fn row_mul_acc(c:&mut [Tile], a:Tile, b:&[Tile]) {
    /* Row operation: c += a*b, where a is a single tile, and (b,c) are vectors */
    let len = min(b.len(),c.len());
    if !a.is_zero() {
        let a_study = Accel::compile_mul_table(a);
        for i in 0..len { c[i] ^= Accel::precomputed_mul(&a_study, b[i]); }
    }
});

accelerated!(pub fn row_mul(a:Tile, b:&mut [Tile]) {
    /* Row operation: b = a*b, where a is a single tile, and b is a vector */
    if a.is_zero() {
        for i in 0..b.len() { b[i] = Tile::ZERO; }
    } else {
        let a_study = Accel::compile_mul_table(a);
        for i in 0..b.len() { b[i] = Accel::precomputed_mul(&a_study, b[i]); }
    }
});

accelerated!(pub fn bulk_swap2_rows(a: &mut [Tile], b: &mut [Tile], ra:Index, rb:Index, nrows: usize) {
    /* Double-row operation: swap rows ra..ra+nrows from a with rb..rb+nrows from b */
    let ab = Tile::IDENTITY.extract_cols(rb,ra,nrows);
    let ba = Tile::IDENTITY.extract_cols(ra,rb,nrows);
    row_mul_acc::with_accel::<Accel>(a,ab,b);
    row_mul_acc::with_accel::<Accel>(b,ba,a);
    row_mul_acc::with_accel::<Accel>(a,ab,b);
});

accelerated!(pub fn bulk_swap_rows(a: &mut [Tile], ra:Index, rb:Index, nrows: usize) {
    /* Single-row operation: swap rows ra..ra+nrows with rb..rb+nrows from b */
    /* Permute columns of the identity; it's equivalent to swapping rows
     * PERF: could probably accelerate making this permutation
     */
    let mut op = Tile::IDENTITY;
    let mut perm = PERMUTE_ALL_ZERO;
    for i in 0..Tile::EDGE_BITS {
        if i >= ra && i < ra+nrows {
            perm[i] = (i-ra+rb) as u8;
        } else if i >= rb && i < rb+nrows {
            perm[i] = (i-rb+ra) as u8;
        } else {
            perm[i] = i as u8;
        }
    }
    Accel::mut_permute_columns(&mut op, &perm);
    let study = Accel::compile_mul_table(op);
    for i in 0..a.len() { a[i] = Accel::precomputed_mul(&study, a[i]); }
});

accelerated!(fn tile_mut_permute_columns(t: &mut Tile, p:&Permutation) {
    /* Permute the columns of t */
    Accel::mut_permute_columns(t,p);
});

accelerated!(pub fn bulk_permute_accum_columns(
    dst: &mut [Tile], dst_stride:usize,
    src: &[Tile], src_stride:usize,
    ntiles: usize,
    p:&Permutation
) {
    if ntiles == 0 { return; }
    assert!((ntiles-1)*dst_stride < dst.len());
    assert!((ntiles-1)*src_stride < src.len());
    for i in 0..ntiles {
        let mut tmp = src[src_stride*i];
        Accel::mut_permute_columns(&mut tmp, p);
        dst[dst_stride*i] += tmp;
    }
});

accelerated!(pub fn bulk_permute_set_columns(
    dst: &mut [Tile], dst_stride:usize,
    src: &[Tile], src_stride:usize,
    ntiles: usize,
    p:&Permutation
) {
    if ntiles == 0 { return; }
    assert!((ntiles-1)*dst_stride < dst.len());
    assert!((ntiles-1)*src_stride < src.len());
    for i in 0..ntiles {
        let mut tmp = src[src_stride*i];
        Accel::mut_permute_columns(&mut tmp, p);
        dst[dst_stride*i] = tmp;
    }
});

/* First column is accum, second is set */
accelerated!(pub fn bulk_permute_accum2_columns(
    dst: &mut [Tile], dst_stride:usize,
    src: &[Tile], src_stride:usize,
    ntiles: usize,
    p0:&Permutation,
    p1:&Permutation
) {
    if ntiles == 0 { return; }
    assert!((ntiles-1)*dst_stride+1 < dst.len());
    assert!((ntiles-1)*src_stride < src.len());
    for i in 0..ntiles {
        /* PERF: do ops like this copy the tiles? */
        let tmp = src[src_stride*i];
        let mut t0 = tmp;
        let mut t1 = tmp;
        Accel::mut_permute_columns(&mut t0, p0);
        Accel::mut_permute_columns(&mut t1, p1);
        dst[dst_stride*i]   += t0;
        dst[dst_stride*i+1]  = t1;
    }
});

/****************************************************************************
 * Scalar "accelerator"
 ****************************************************************************/
pub struct Scalar;
impl Accelerator for Scalar {
    type MulTable = Tile;

    #[inline(always)]
    fn is_available() -> bool { true }

    // Unsafe because: Actually safe, but the trait sig is unsafe
    #[inline(always)]
    unsafe fn mut_permute_columns(a:&mut Tile, permutation:&Permutation) {
        let mut ret = Tile::ZERO;
        for i in 0..Tile::EDGE_BITS {
            if permutation[i] != PERMUTE_ZERO {
                ret.mut_xor_col(a,i,permutation[i] as Index);
            }
        }
        *a = ret;
    }

    #[inline(always)]
    // Unsafe because: Actually safe, but the trait sig is unsafe
    unsafe fn compile_mul_table(t: Tile) -> Self::MulTable { t }

    #[inline(always)]
    // Unsafe because: Actually safe, but the trait sig is unsafe
    unsafe fn precomputed_mul(table: &Self::MulTable, t: Tile) -> Tile { *table*t }
}

/****************************************************************************
 * Intel AVX2 acceleration
 ****************************************************************************/

/**************************************************************************
 * Vectorized multiplication strategy:
 *
 * We assume that the compiler is smart enough to vectorize most operations,
 * but here we use a variant of the Method of the Four Russians which is too
 * tricky for the compiler to figure out.
 * 
 * The idea is to consider an 8x8 subtile L*R operation, where each
 * byte of the output depends on only one byte of the R.  For any given L,
 * this is a linear 8->8 function, which can be decomposed as two 4->8
 * on the nibbles.  These can be computed using two vector permute operations.
 **************************************************************************/

#[cfg(target_arch="x86_64")]
pub mod avx2 {
    use super::*;
    #[cfg(target_arch="x86_64")]
    use core::arch::x86_64::*;
    #[cfg(target_arch="x86")]
    use core::arch::x86::*;
    pub struct Avx2;

    #[derive(Clone,Copy)]
    pub struct MulTable { table : [__m256i; 4] }

    impl Accelerator for Avx2 {
        type MulTable = MulTable;

        #[inline(always)]
        fn is_available() -> bool { is_x86_feature_detected!("avx2") }

        /// "Permute" columns of the tile according to "permutation".
        ///
        /// New column x = old column permutation(x).
        /// ("permutation" need not actually be a permutation)
        /// Any value greater than 0xF (in particular, PERMUTE_ZERO)
        /// will result in the column becoming zero.
        /// 
        /// PERF: adding a permute2 would improve performance for certain column
        /// ops on neon, but possibly not on AVX2
        ///
        /// Unsafe because: uses AVX2, will catch SIGILL if not supported.
        #[inline(always)]
        unsafe fn mut_permute_columns(t:&mut Tile, permutation:&Permutation) {
            let addr = permutation as *const u8 as *const __m128i;
            let vperm = _mm256_loadu2_m128i(addr,addr);
            let ab = _mm256_loadu_si256(&t.storage[0] as *const u64 as *const __m256i);
            let ab = _mm256_shuffle_epi8(ab,vperm);
            _mm256_storeu_si256(&mut t.storage[0] as *mut u64 as *mut __m256i, ab);
        }

        /// Precompute multiples of a tile in order to speed up vectorized multiplication
        ///
        /// Unsafe because: uses AVX2, will catch SIGILL if not supported.
        #[inline(always)]
        unsafe fn compile_mul_table(t:Tile) -> MulTable {
            let mut abcd = _mm256_loadu_si256(&t.storage[0] as *const u64 as *const __m256i);
            let index = _mm256_set_epi64x(0x0F0E0D0C0B0A0908,0x0706050403020100,
                                            0x0F0E0D0C0B0A0908,0x0706050403020100);
            let lane0 = _mm256_setzero_si256();
            let lane4 = _mm256_set1_epi8(4);
            let lane8 = _mm256_set1_epi8(8);
            let lane12 = _mm256_set1_epi8(12);
            let mut one  = _mm256_set1_epi8(1);
            let mut aclo = _mm256_setzero_si256();
            let mut achi = _mm256_setzero_si256();
            let mut bdlo = _mm256_setzero_si256();
            let mut bdhi = _mm256_setzero_si256();
            for _ in 0..4 {
                let tlo = _mm256_cmpeq_epi8(_mm256_and_si256(index,  one), one);
                one  = _mm256_slli_epi16(one,1);
                aclo = _mm256_xor_si256(aclo,  _mm256_and_si256(_mm256_shuffle_epi8(abcd,lane0),tlo));
                achi = _mm256_xor_si256(achi,  _mm256_and_si256(_mm256_shuffle_epi8(abcd,lane4),tlo));
                bdlo = _mm256_xor_si256(bdlo,  _mm256_and_si256(_mm256_shuffle_epi8(abcd,lane8),tlo));
                bdhi = _mm256_xor_si256(bdhi,  _mm256_and_si256(_mm256_shuffle_epi8(abcd,lane12),tlo));
                abcd = _mm256_srli_epi64(abcd,8);
            }
            let adlo = _mm256_permute2x128_si256(aclo,bdlo,0x30);
            let adhi = _mm256_permute2x128_si256(achi,bdhi,0x30);
            let cblo = _mm256_permute2x128_si256(aclo,bdlo,0x21);
            let cbhi = _mm256_permute2x128_si256(achi,bdhi,0x21);
            MulTable { table : [ adlo, adhi, cblo, cbhi ] }
        }

        /// Use precomputed table in order to speed up vectorized multiplication
        ///
        /// Unsafe because: uses AVX2, will catch SIGILL if not supported.
        #[inline(always)]
        unsafe fn precomputed_mul(table: &MulTable, tv: Tile) -> Tile {
            let mut ret = [0u64; STORAGE_PER];
            let low_nibble = _mm256_set1_epi8(0xF);
            let vec = _mm256_loadu_si256(&tv.storage[0] as *const u64 as  *const __m256i);
            let veclo = _mm256_and_si256(vec, low_nibble);
            let vechi = _mm256_and_si256(_mm256_srli_epi16(vec,4), low_nibble);
            let [adlo,adhi,cblo,cbhi] = table.table;
            let ad = _mm256_xor_si256(_mm256_shuffle_epi8(adlo, veclo),
                                        _mm256_shuffle_epi8(adhi, vechi));
            let cb = _mm256_xor_si256(_mm256_shuffle_epi8(cblo, veclo),
                                        _mm256_shuffle_epi8(cbhi, vechi));
            let vret = _mm256_xor_si256(_mm256_permute2x128_si256(cb,cb,1), ad);
            _mm256_storeu_si256(&mut ret[0] as *mut u64 as *mut __m256i, vret);
            Tile { storage : ret }
        }
    }
}

/****************************************************************************
 * ARM NEON acceleration
 ****************************************************************************/

#[cfg(any(target_arch="aarch64",target_arch="armv7"))]
pub mod neon {
    #[cfg(target_arch="aarch64")]
    use core::arch::aarch64::*;

    #[cfg(target_arch="armv7")]
    use core::arch::arm::*;
    use super::*;

    pub struct Neon;

    #[derive(Clone,Copy)]
    pub struct MulTable { table : [(uint8x16_t,uint8x16_t); 4] }

    impl Accelerator for Neon {
        type MulTable = MulTable;

        /** As far as I know, NEON is mandatory on aarch64 */
        #[cfg(target_arch="aarch64")]
        #[inline(always)]
        fn is_available() -> bool { true }

        #[cfg(target_arch="armv7")]
        #[inline(always)]
        fn is_available() -> bool { is_arm_feature_detected("neon") }

        /// "Permute" columns of the tile according to "permutation".
        ///
        /// New column x = old column permutation(x).
        /// ("permutation" need not actually be a permutation)
        /// Any value greater than 0xF (in particular, PERMUTE_ZERO)
        /// will result in the column becoming zero.
        /// 
        /// PERF: adding a permute2 would improve performance for certain column
        /// ops on neon, but possibly not on AVX2
        ///
        /// Unsafe because: uses NEON intrinsics (unsafe at least on Armv7, which may not have NEON)
        #[inline(always)]
        unsafe fn mut_permute_columns(t:&mut Tile, permutation:&Permutation) {
            let vperm = vld1q_u8(permutation as *const u8);
            let ab = vreinterpretq_u8_u64(vld1q_u64(&t.storage[0] as *const u64));
            let cd = vreinterpretq_u8_u64(vld1q_u64(&t.storage[2] as *const u64));
            let ab = vqtbl1q_u8(ab, vperm);
            let cd = vqtbl1q_u8(cd, vperm);
            vst1q_u64(&mut t.storage[0] as *mut u64, vreinterpretq_u64_u8(ab));
            vst1q_u64(&mut t.storage[2] as *mut u64, vreinterpretq_u64_u8(cd));
        }

        /// Precompute multiples of a tile in order to speed up vectorized multiplication
        //
        /// Unsafe because: uses NEON intrinsics (unsafe at least on Armv7, which may not have NEON)
        #[inline(always)]
        unsafe fn compile_mul_table(t:Tile) -> MulTable {
            let mut ab = vld1q_u64(&t.storage[0] as *const u64);
            let mut cd = vld1q_u64(&t.storage[2] as *const u64);
            let index_low = vcombine_u8(vcreate_u8(0x0706050403020100),vcreate_u8(0x0F0E0D0C0B0A0908));
            let mut one  = vdupq_n_u8(1);
            let mut low0  = vdupq_n_u8(0);
            let mut high0 = vdupq_n_u8(0);
            let mut low1  = vdupq_n_u8(0);
            let mut high1 = vdupq_n_u8(0);
            let mut low2  = vdupq_n_u8(0);
            let mut high2 = vdupq_n_u8(0);
            let mut low3  = vdupq_n_u8(0);
            let mut high3 = vdupq_n_u8(0);
            for _ in 0..4 {
                let tlo = vceqq_u8(vandq_u8(index_low,  one), one);
                one = vshlq_n_u8(one,1);
                low0  = veorq_u8(low0,  vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(ab),0),tlo));
                high0 = veorq_u8(high0, vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(ab),4),tlo));
                low1  = veorq_u8(low1,  vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(ab),8),tlo));
                high1 = veorq_u8(high1, vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(ab),12),tlo));
                ab = vshrq_n_u64(ab,8);
                low2  = veorq_u8(low2,  vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(cd),0),tlo));
                high2 = veorq_u8(high2, vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(cd),4),tlo));
                low3  = veorq_u8(low3,  vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(cd),8),tlo));
                high3 = veorq_u8(high3, vandq_u8(vdupq_laneq_u8(vreinterpretq_u8_u64(cd),12),tlo));
                cd = vshrq_n_u64(cd,8);
            }
            MulTable { table : [ (low0,high0),(low1,high1),(low2,high2),(low3,high3) ] }
        }

        /// Use precomputed table in order to speed up vectorized multiplication
        ///
        /// Unsafe because: uses NEON intrinsics (unsafe at least on Armv7, which may not have NEON)
        #[inline(always)]
        unsafe fn precomputed_mul(table: &MulTable, tv: Tile) -> Tile {
            let mut ret = [0u64; STORAGE_PER];
            let low_nibble = vdupq_n_u8(0xF);
            let ab = vreinterpretq_u8_u64(vld1q_u64(&tv.storage[0] as *const u64));
            let cd = vreinterpretq_u8_u64(vld1q_u64(&tv.storage[2] as *const u64));
            let [(elo,ehi),(flo,fhi),(glo,ghi),(hlo,hhi)] = table.table;
            let ij = veorq_u8(vqtbl1q_u8(elo, vandq_u8(ab,low_nibble)),
                                vqtbl1q_u8(ehi, vshrq_n_u8(ab,4)));
            let ij = veorq_u8(ij,
                        veorq_u8(vqtbl1q_u8(flo, vandq_u8(cd,low_nibble)),
                                vqtbl1q_u8(fhi, vshrq_n_u8(cd,4))));
            let kl = veorq_u8(vqtbl1q_u8(glo, vandq_u8(ab,low_nibble)),
                                vqtbl1q_u8(ghi, vshrq_n_u8(ab,4)));
            let kl = veorq_u8(kl,
                        veorq_u8(vqtbl1q_u8(hlo, vandq_u8(cd,low_nibble)),
                                vqtbl1q_u8(hhi, vshrq_n_u8(cd,4))));
            vst1q_u64(&mut ret[0] as *mut u64, vreinterpretq_u64_u8(ij));
            vst1q_u64(&mut ret[2] as *mut u64, vreinterpretq_u64_u8(kl));
            Tile { storage : ret }
        }
    }
}

/**************************************************************************
 * Tests
 **************************************************************************/
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng,thread_rng};

    fn random_tile() -> Tile { thread_rng().gen::<Tile>() }

    /** Test some properties of single-bit matrices */
    #[test]
    fn single_bit_tests() {
        for _ in 0..100 {
            let i = thread_rng().gen_range(0..8);
            let j = thread_rng().gen_range(0..8);
            let k = thread_rng().gen_range(0..8);
            assert_eq!(Tile::single_bit(i,j) * Tile::single_bit(j,k), Tile::single_bit(i,k));
            assert_eq!(Tile::single_bit(i,j) * Tile::single_bit(j,k), Tile::single_bit(i,k));
            assert_eq!(Tile::single_bit(i,j).get_bit(i,j), true);
            for k in 0..Tile::EDGE_BITS {
                assert_eq!(Tile::single_bit(i,j).first_nonzero_entry_in_col(k,!0), if j==k { Some(i) } else { None });
                assert_eq!(Tile::ZERO.first_nonzero_entry_in_col(k,!0), None);
            }
        }
    }

    /** Test that addition commutes and associates, and is the same as xor */
    #[test]
    fn add_identities() {
        for _ in 0..100 {
            let t = random_tile();
            let u = random_tile();
            let v = random_tile();
            assert_eq!(t + Tile::ZERO, t);
            assert_eq!(Tile::ZERO + t, t);
            assert_eq!(t + t, Tile::ZERO);
            assert_eq!(t + u, u + t);
            assert_eq!(t + u, t ^ u);
            assert_eq!((t + u) + v, t + (u + v));
        }
    }

    /** Test that multiplication associates, and gets flipped by transpose */
    #[test]
    fn mul_identities() {
        for _ in 0..100 {
            let t = random_tile();
            let u = random_tile();
            let v = random_tile();
            assert_eq!(t * Tile::IDENTITY, t);
            assert_eq!(Tile::IDENTITY * t, t);
            assert_eq!((t * u) * v, t * (u * v));
            assert_eq!((t * u).transpose(), u.transpose() * t.transpose());
        }
    }

    /** Test that transpose and multiply distribute over add */
    #[test]
    fn linearity() {
        for _ in 0..100 {
            let t = random_tile();
            let u = random_tile();
            let v = random_tile();
            let tt = t.transpose();
            assert_eq!(t,t.transpose().transpose());
            for row in 0..Tile::EDGE_BITS {
                for col in 0..Tile::EDGE_BITS {
                    assert_eq!(tt.get_bit(row,col), t.get_bit(col,row));
                }
            }
            assert_eq!(t*(u+v), t*u + t*v);
            assert_eq!((u+v)*t, u*t + v*t);
            assert_eq!((t+u).transpose(), t.transpose()+u.transpose());
        }
    }

    /** Test precomputed vectorized multiplication */
    #[test]
    fn vectorized() {
        for _ in 0..100 {
            let a = random_tile();
            let t = random_tile();
            let u = [random_tile()];
            let mut v = [a];
            row_mul_acc::runtime(&mut v, t, &u);
            assert_eq!(v[0], a+t*u[0]);
        }
    }

    /** Test that row operations work */
    #[test]
    fn rowops() {
        for length in 0..100 {
            let t = random_tile();
            let mut ipt : Vec<Tile> = vec![Tile::ZERO; length];
            let mut pre : Vec<Tile> = vec![Tile::ZERO; length];
            let mut opt : Vec<Tile> = vec![Tile::ZERO; length];

            for i in 0..length { 
                ipt[i] = random_tile();
                pre[i] = random_tile();
                opt[i] = pre[i];
            }
            row_mul_acc::runtime(&mut opt,t,&ipt);
            for i in 0..length {
                assert_eq!(opt[i], pre[i]+t*ipt[i]);
            }

            opt = ipt.clone();
            row_mul::runtime(t,&mut opt);
            for i in 0..length {
                assert_eq!(opt[i], t*ipt[i]);
            }
        }
    }

    /** Test properties of the pseudoinverse */
    #[test]
    fn pseudoinverse() {
        for _ in 0..100 {
            let t     = random_tile();
            let avail = thread_rng().gen::<Edge>();
            let (psi,perm,ech) = t.pseudoinverse(avail);
            let mut prod = psi * t;
            prod.mut_permute_columns(&perm);
            assert_eq!(prod & Tile::IDENTITY, prod);
            for i in 0..Tile::EDGE_BITS {
                let x = perm[i];
                if x == PERMUTE_ZERO { continue; }
                assert!((avail>>i & 1) != 0);
                assert!((ech>>x & 1) != 0);
            }
        }
    }

    /** Test that broadcast and mask work in reasonable ways */
    #[test]
    fn broadcast_and_mask() {
        for i in 0..Tile::EDGE_BITS {
            assert_eq!(Tile::row_mask(i), Tile::rows_mask(i,1));
            assert_eq!(Tile::col_mask(i), Tile::cols_mask(i,1));
            for j in i..=Tile::EDGE_BITS {
                let e = thread_rng().gen::<Edge>();
                let mr = Tile::rows_mask(i,j-i);
                let mc = Tile::cols_mask(i,j-i);
                let br = Tile::broadcast_edge_as_row(e);
                let bc = Tile::broadcast_edge_as_col(e);
                for row in 0..Tile::EDGE_BITS {
                    for col in 0..Tile::EDGE_BITS {
                        assert_eq!(mr.get_bit(row,col), row >= i && row < j);
                        assert_eq!(mc.get_bit(row,col), col >= i && col < j);
                        assert_eq!(br.get_bit(row,col), (e>>col & 1) != 0);
                        assert_eq!(bc.get_bit(row,col), (e>>row & 1) != 0);
                    }
                }
            }
        }
    }

    /** Test permutations of columns */
    #[test]
    fn perumations() {
        for _ in 0..100 {
            let t = random_tile();
            let mut p = PERMUTE_ALL_ZERO;
            let mut q = PERMUTE_ALL_ZERO;
            let mut pmatrix = Tile::ZERO;
            for i in 0..Tile::EDGE_BITS {
                if thread_rng().gen::<bool>() {
                    let tmp = thread_rng().gen_range(0..Tile::EDGE_BITS);
                    p[i] = tmp as u8;
                    pmatrix.set_bit(tmp,i);
                };
                if thread_rng().gen::<bool>() {
                    let tmp = thread_rng().gen_range(0..Tile::EDGE_BITS) as u8;
                    q[i] = tmp as u8;
                };
                assert_eq!(t*&p, t*pmatrix);
            }
        }
    }

    /* TEST: test set row */
}
