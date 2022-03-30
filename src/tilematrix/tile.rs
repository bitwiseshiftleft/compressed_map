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

    /** Permute our columns in place */
    pub fn mut_permute_columns(&mut self, perm: &Permutation) {
        #[cfg(target_arch="x86_64")]
        if vectorized_avx2::is_available() {
            vectorized_avx2::mut_permute_columns(self,perm);
            return ();
        }

        #[cfg(target_arch="aarch64")]
        if vectorized_neon::is_available() {
            vectorized_neon::mut_permute_columns(self,perm);
            return ();
        }

        scalar_core::mut_permute_columns(self,perm);
    }

    /** Permute our columns out of place */
    pub fn permute_columns(self, perm: &Permutation) -> Self {
        let mut ret = self;
        ret.mut_permute_columns(perm);
        ret
    }

    /** Permute our rows out of place */
    pub fn permute_rows(self, perm: &Permutation) -> Self {
        self.transpose().permute_columns(perm).transpose()
    }

    /** Compose permutations */
    pub fn compose_permutations(perm1: &Permutation, perm2: &Permutation) -> Permutation {
        #[cfg(target_arch="x86_64")]
        if vectorized_avx2::is_available() {
            return vectorized_avx2::compose_permutations(perm1,perm2);
        }

        #[cfg(target_arch="aarch64")]
        if vectorized_neon::is_available() {
            return vectorized_neon::compose_permutations(perm1,perm2);
        }

        scalar_core::compose_permutations(perm1,perm2)
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

    /** Swap col a..a+ncols and cols b..b+ncols */
    pub fn mut_swap_cols(&mut self, cola:Index, colb:Index, ncols:usize) {
        let mut perm = PERMUTE_ALL_ZERO;
        for i in 0..Tile::EDGE_BITS {
            if i >= cola && i < cola+ncols {
                perm[i] = (i-cola+colb) as u8;
            } else if i >= colb && i < colb+ncols {
                perm[i] = (i-colb+cola) as u8;
            } else {
                perm[i] = i as u8;
            }
        }
        self.mut_permute_columns(&perm);
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

/**************************************************************************
 * Vectorized multiplication.
 * We assume that the compiler is smart enough to vectorize most operations,
 * but here we use a variant of the Method of the Four Russians which is too
 * tricky for the compiler to figure out.
 * 
 * The idea is to consider an 8x8 subtile L*R operation, where each
 * byte of the output depends on only one byte of the R.  For any given L,
 * this is a linear 8->8 function, which can be decomposed as two 4->8
 * on the nibbles.  These can be computed using two vector permute operations.
 * 
 * TODO: feature gate intrinsic versions
 **************************************************************************/
mod scalar_core {
    use crate::tile::{Tile,Permutation,PERMUTE_ZERO,PERMUTE_ALL_ZERO,Index};
    pub fn mut_permute_columns(a:&mut Tile, permutation:&Permutation) {
        let mut ret = Tile::ZERO;
        for i in 0..Tile::EDGE_BITS {
            if permutation[i] != PERMUTE_ZERO {
                ret.mut_xor_col(a,i,permutation[i] as Index);
            }
        }
        *a = ret;
    }
    pub fn compose_permutations(perm1:&Permutation, perm2:&Permutation) -> Permutation {
        let mut ret = PERMUTE_ALL_ZERO;
        for i in 0..Tile::EDGE_BITS {
            if perm1[i] != PERMUTE_ZERO {
                ret[i] = perm2[perm1[i] as usize];
            }
        }
        ret
    }
}


#[cfg(target_arch="x86_64")]
mod vectorized_avx2 {
    use std::ops::Mul;
    use crate::tile::{Tile,Permutation,PERMUTE_ZERO,PERMUTE_ALL_ZERO,STORAGE_PER};
    use core::arch::x86_64::*;

    #[derive(Clone,Copy)]
    pub struct MulTable {
        table : [__m256i; 4]
    }

    #[inline(always)]
    pub fn is_available() -> bool { true /*is_x86_feature_detected!("avx2")*/ }

    /** "Permute" columns of the tile according to "permutation".
     * New column x = old column permutation(x).
     * ("permutation" need not actually be a permutation)
     * Any value greater than 0xF (in particular, PERMUTE_ZERO)
     * will result in the column becoming zero.
     * 
     * PERF: adding a permute2 would improve performance for certain column
     * ops on neon, but possibly not on AVX2
     */
    #[inline(always)]
    pub fn mut_permute_columns(t:&mut Tile, permutation:&Permutation) {
        unsafe {
            let addr = permutation as *const u8 as *const __m128i;
            let vperm = _mm256_loadu2_m128i(addr,addr);
            let ab = _mm256_loadu_si256(&t.storage[0] as *const u64 as *const __m256i);
            let ab = _mm256_shuffle_epi8(ab,vperm);
            _mm256_storeu_si256(&mut t.storage[0] as *mut u64 as *mut __m256i, ab);
        }
    }

    #[inline(always)]
    pub fn compose_permutations(perm1:&Permutation, perm2:&Permutation) -> Permutation {
        unsafe {
            let mut ret:Permutation = PERMUTE_ALL_ZERO;
            let z = _mm_set1_epi8(PERMUTE_ZERO as i8);
            let vperm1 = _mm_loadu_si128(perm1 as *const u8 as *const __m128i);
            let vperm2 = _mm_loadu_si128(perm2 as *const u8 as *const __m128i);
            let vperm12 = _mm_xor_si128(z,_mm_shuffle_epi8(_mm_xor_si128(z,vperm2),vperm1));
            _mm_storeu_si128(&mut ret[0] as *mut u8 as *mut __m128i, vperm12);
            ret
        }
    }

    /** Precompute multiples of a tile in order to speed up vectorized multiplication */
    #[inline(always)]
    pub fn compile_mul_table(t:Tile) -> MulTable {
        unsafe {
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
    }

    impl Mul<Tile> for MulTable {
        type Output = Tile;
        #[inline(always)]
        fn mul(self, tv: Tile) -> Tile {
            let mut ret = [0u64; STORAGE_PER];
            unsafe {
                let low_nibble = _mm256_set1_epi8(0xF);
                let vec = _mm256_loadu_si256(&tv.storage[0] as *const u64 as  *const __m256i);
                let veclo = _mm256_and_si256(vec, low_nibble);
                let vechi = _mm256_and_si256(_mm256_srli_epi16(vec,4), low_nibble);
                let [adlo,adhi,cblo,cbhi] = self.table;
                let ad = _mm256_xor_si256(_mm256_shuffle_epi8(adlo, veclo),
                                          _mm256_shuffle_epi8(adhi, vechi));
                let cb = _mm256_xor_si256(_mm256_shuffle_epi8(cblo, veclo),
                                          _mm256_shuffle_epi8(cbhi, vechi));
                let vret = _mm256_xor_si256(_mm256_permute2x128_si256(cb,cb,1), ad);
                _mm256_storeu_si256(&mut ret[0] as *mut u64 as *mut __m256i, vret);
            }
            Tile { storage : ret }
        }
    }
}

#[cfg(all(target_arch="aarch64"))]
mod vectorized_neon {
    use std::ops::Mul;
    use crate::tile::{Tile,Permutation,PERMUTE_ZERO,PERMUTE_ALL_ZERO,STORAGE_PER};
    use core::arch::aarch64::*;

    #[derive(Clone,Copy)]
    pub struct MulTable {
        table : [(uint8x16_t,uint8x16_t); 4]
    }

    /** As far as I know, NEON is mandatory on aarch64 */
    #[inline(always)]
    pub fn is_available() -> bool { true }

    /** "Permute" columns of the tile according to "permutation".
     * New column x = old column permutation(x).
     * ("permutation" need not actually be a permutation)
     * Any value greater than 0xF (in particular, PERMUTE_ZERO)
     * will result in the column becoming zero.
     * 
     * PERF: adding a permute2 would improve performance for certain column
     * ops on neon, but possibly not on AVX2
     */
    pub fn mut_permute_columns(t:&mut Tile, permutation:&Permutation) {
        unsafe {
            let vperm = vld1q_u8(permutation as *const u8);
            let ab = vreinterpretq_u8_u64(vld1q_u64(&t.storage[0] as *const u64));
            let cd = vreinterpretq_u8_u64(vld1q_u64(&t.storage[2] as *const u64));
            let ab = vqtbl1q_u8(ab, vperm);
            let cd = vqtbl1q_u8(cd, vperm);
            vst1q_u64(&mut t.storage[0] as *mut u64, vreinterpretq_u64_u8(ab));
            vst1q_u64(&mut t.storage[2] as *mut u64, vreinterpretq_u64_u8(cd));
        }
    }

    pub fn compose_permutations(perm1:&Permutation, perm2:&Permutation) -> Permutation {
        unsafe {
            let mut ret:Permutation = PERMUTE_ALL_ZERO;
            let vall   = vdupq_n_u8(PERMUTE_ZERO);
            let vperm1 = vld1q_u8(perm1 as *const u8);
            let vperm2 = vld1q_u8(perm2 as *const u8);
            vst1q_u8(&mut ret[0] as *mut u8, veorq_u8(vall,vqtbl1q_u8(veorq_u8(vall,vperm2),vperm1)));
            ret
        }
    }

    /** Precompute multiples of a tile in order to speed up vectorized multiplication */
    pub fn compile_mul_table(t:Tile) -> MulTable {
        unsafe {
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
    }

    impl Mul<Tile> for MulTable {
        type Output = Tile;
        fn mul(self, tv: Tile) -> Tile {
            let mut ret = [0u64; STORAGE_PER];
            unsafe {
                let low_nibble = vdupq_n_u8(0xF);
                let ab = vreinterpretq_u8_u64(vld1q_u64(&tv.storage[0] as *const u64));
                let cd = vreinterpretq_u8_u64(vld1q_u64(&tv.storage[2] as *const u64));
                let [(elo,ehi),(flo,fhi),(glo,ghi),(hlo,hhi)] = self.table;
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
            }
            Tile { storage : ret }
        }
    }
}

/**************************************************************************
 * Vectorized row operations
 **************************************************************************/

/** Row operation: c += a*b, where a is a single tile, and (b,c) are vectors */
pub fn row_mul_acc(c:&mut [Tile], a:Tile, b:&[Tile]) {
    let len = min(b.len(),c.len());
    if !a.is_zero() {
        #[cfg(target_arch="x86_64")]
        if vectorized_avx2::is_available() {
            let a_study = vectorized_avx2::compile_mul_table(a);
            for i in 0..len { c[i] ^= a_study * b[i]; }
            return ();
        }

        #[cfg(target_arch="aarch64")]
        if vectorized_neon::is_available() {
            let a_study = vectorized_neon::compile_mul_table(a);
            for i in 0..len { c[i] ^= a_study * b[i]; }
            return ();
        }

        for i in 0..len { c[i] ^= a * b[i]; }
    }
}

/** Row operation: b = a*b, where a is a single tile, and b is a vector */
pub fn row_mul(a:Tile, b:&mut [Tile]) {
    if a.is_zero() {
        for i in 0..b.len() { b[i] = Tile::ZERO; }
    } else {
        #[cfg(target_arch="x86_64")]
        if vectorized_avx2::is_available() {
            let a_study = vectorized_avx2::compile_mul_table(a);
            for i in 0..b.len() { b[i] = a_study * b[i]; }
            return ();
        }

        #[cfg(target_arch="aarch64")]
        if vectorized_neon::is_available() {
            let a_study = vectorized_neon::compile_mul_table(a);
            for i in 0..b.len() { b[i] = a_study * b[i]; }
            return ();
        }

        for i in 0..b.len() { b[i] = a * b[i]; }
    }
}

/** Double-row operation: swap rows ra..ra+nrows from a with rb..rb+nrows from b */
pub fn bulk_swap2_rows(a: &mut [Tile], b: &mut [Tile], ra:Index, rb:Index, nrows: usize) {
    let ab = Tile::IDENTITY.extract_cols(rb,ra,nrows);
    let ba = Tile::IDENTITY.extract_cols(ra,rb,nrows);

    row_mul_acc(a,ab,b);
    row_mul_acc(b,ba,a);
    row_mul_acc(a,ab,b);
}

/** Single-row operation: swap rows ra..ra+nrows with rb..rb+nrows from b */
pub fn bulk_swap_rows(a: &mut [Tile], ra:Index, rb:Index, nrows: usize) {
    let mut op = Tile::IDENTITY;
    op.mut_swap_cols(rb,ra,nrows); // swapping rows of id is the same as swapping columns

    #[cfg(target_arch="x86_64")]
    if vectorized_avx2::is_available() {
        let study = vectorized_avx2::compile_mul_table(op);
        for i in 0..a.len() { a[i] = study * a[i]; }
        return ();
    }

    #[cfg(target_arch="aarch64")]
    if vectorized_neon::is_available() {
        let study = vectorized_neon::compile_mul_table(op);
        for i in 0..a.len() { a[i] = study * a[i]; }
        return ();
    }

    for i in 0..a.len() { a[i] = op * a[i]; }
}

/**************************************************************************
 * Tests
 **************************************************************************/
#[cfg(test)]
mod tests {
    use crate::tile::{Edge,Tile,row_mul,row_mul_acc,PERMUTE_ZERO,PERMUTE_ALL_ZERO,scalar_core};
    #[cfg(target_arch="x86_64")]
    use crate::tile::vectorized_avx2;
    #[cfg(target_arch="aarch64")]
    use crate::tile::vectorized_neon;
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
            #[cfg(target_arch="x86_64")]
            if vectorized_avx2::is_available() {
                let t = random_tile();
                let u = random_tile();
                let s = vectorized_avx2::compile_mul_table(t);
                assert_eq!(s*u,t*u);
            }

            #[cfg(target_arch="aarch64")]
            if vectorized_neon::is_available() {
                let t = random_tile();
                let u = random_tile();
                let s = vectorized_neon::compile_mul_table(t);
                assert_eq!(s*u,t*u);
            }
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
            row_mul_acc(&mut opt,t,&ipt);
            for i in 0..length {
                assert_eq!(opt[i], pre[i]+t*ipt[i]);
            }

            opt = ipt.clone();
            row_mul(t,&mut opt);
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
            scalar_core::mut_permute_columns(&mut prod, &perm);
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


                #[cfg(target_arch="x86_64")]
                if vectorized_avx2::is_available() {
                    assert_eq!((t*&p)*&q, t*&vectorized_avx2::compose_permutations(&q,&p));
                }

                #[cfg(target_arch="aarch64")]
                if vectorized_neon::is_available() {
                    assert_eq!((t*&p)*&q, t*&vectorized_neon::compose_permutations(&q,&p));
                }
                assert_eq!((t*&p)*&q, t*&scalar_core::compose_permutations(&q,&p));
            }
        }
    }

    /* TODO: test set row */
}
