/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Uniform sparse linear map implementation.
 */

use crate::tilematrix::matrix::{Matrix,Systematic};
use crate::tilematrix::bitset::BitSet;
use std::marker::PhantomData;
use core::hash::Hash;
use std::cmp::max;
use rand::{RngCore};
use rand::rngs::OsRng;
use std::mem::replace;
use siphasher::sip128::{Hasher128, SipHasher13, Hash128};

/** Space of responses in dictionary impl. */
pub type Response = u64;

type LfrHasher = SipHasher13;

/** A key for the SipHash13 hash function. */
pub type HasherKey = [u8; 16];

type RowIdx   = u64; // if u32 then limit to ~4b rows
type BlockIdx = u32;
type LfrBlock = u32;

/** The internal block size of the map, in bytes. */
pub const BLOCKSIZE : usize = (LfrBlock::BITS as usize) / 8;
const OVERPROVISION : u64 = /* 1/ */ 1024;
const EXTRA_ROWS : usize = 8;

/** Result of hashing an item */
struct LfrRow {
    block_positions : [BlockIdx; 2],
    block_key       : [LfrBlock; 2],
    augmented       : Response
}

/** Domain separator for the hash */
#[derive(Hash)]
enum WhyHashing {
    HashingInput,
    DerivingNewKey,
    RandomizingSolution
}

/**
 * The main sampling function: given a seed, sample block positions.
 * This routine is what characterizes fringe matrices.
 */
fn sample_block_positions(
    mut stride_seed : u64, // only 32 bits used
    a_seed  : u64,         // only 32 bits used
    nblocks : u64
) -> [BlockIdx; 2] {
    debug_assert!(nblocks >= 2);
    let nblocks_huge    = nblocks as u128;

    /* calculate log(nblocks)<<48 in a smooth way */
    let k = 63-nblocks.leading_zeros() as u64;
    let smoothlog = (k<<48) + (nblocks<<(48-k)) - (1<<48);
    let leading_coefficient = (12u64<<8) / BLOCKSIZE as u64; // experimentally determined
    let num = smoothlog * leading_coefficient; // With BLOCKSIZE=4, can't overflow until k=85 which is impossible

    stride_seed |= (1u64<<33) / OVERPROVISION; // | instead of + because it can't overflow
    let den = (((stride_seed * stride_seed) as u128 * nblocks_huge) >> 32) as u64; // can't overflow because stride_seed is only 32 bits
    let b_seed = (num / den + a_seed) & 0xFFFFFFFF; // den can't be 0 because stride_seed is adjusted

    let     a = ((a_seed as u128 * nblocks_huge)>>32) as BlockIdx;
    let mut b = ((b_seed as u128 * nblocks_huge)>>32) as BlockIdx;
    if a==b {
        b += 1;
        if b >= nblocks as BlockIdx {
            b = 0;
        }
    }
    [a,b]
}

/** Interpret a hash as a row */
fn interpret_hash_as_row(hash: Hash128, nblocks:usize) -> LfrRow {
    let stride_seed = hash.h1 & 0xFFFFFFFF;
    let a_seed = hash.h1 >> 32;
    let block_key = [(hash.h2 & 0xFFFFFFFF) as u32, (hash.h2 >> 32) as u32];
    let augmented = hash.h1.wrapping_add(hash.h2).rotate_left(39) ^ hash.h1;
    let block_positions = sample_block_positions(stride_seed, a_seed, nblocks as u64);
    LfrRow { block_positions:block_positions, block_key:block_key, augmented:augmented }
}

/** The outer-main hash function: hash an object to a LfrRow */
fn hash_object_to_row<T:Hash> (key: &HasherKey, nblocks:usize, t: &T)-> LfrRow {
    let mut h = LfrHasher::new_with_key(&key);
    WhyHashing::HashingInput.hash(&mut h);
    t.hash(&mut h);
    interpret_hash_as_row(h.finish128(), nblocks)
}

/** A block or group of blocks for the hierarchical solver */
struct BlockGroup {
    contents:   Matrix,
    solution:   Systematic,
    empty:      bool,
    expected_rows: usize,
    rowids: Vec<(u8,RowIdx)>
}

impl BlockGroup {
    fn new(rows: usize, cols: usize, aug: usize, empty:bool) -> BlockGroup {
        let mut rowids = vec![(0,0);rows+1];
        rowids[rows] = (u8::MAX,RowIdx::MAX);
        BlockGroup {
            contents: Matrix::new(rows, cols, aug),
            solution: Systematic::identity(0),
            empty: empty,
            expected_rows: 0, // set later
            rowids: rowids
        }
    }
}

/**
 * Forward solve step.
 * Merge the resolvable rows of the left and right BlockGroups, then
 * project them out of the remaining ones.  Clears the contents of left and right,
 * but not the solution.
 *
 * Fails if the block group doesn't have full row-rank.
 */
fn forward_solve(left:&mut BlockGroup, right:&mut BlockGroup, nmerge:usize) -> Option<BlockGroup> {
    if right.empty {
        return Some(replace(left, BlockGroup::new(0,0,0,true)));
    } else if left.empty {
        return Some(replace(right, BlockGroup::new(0,0,0,true)));
    }

    /* Prepare to interleave the rows by ID.  Do this by basically mergesort */
    let nrows_out = left.contents.rows + right.contents.rows - 2*nmerge;
    let mut rowids = vec![(0,0); nrows_out+1];
    rowids[nrows_out] = (u8::MAX,RowIdx::MAX);
    let mut interleave_left = BitSet::with_capacity(nrows_out);
    let (mut l,mut r) = (nmerge,nmerge);
    for rowout in 0..nrows_out {
        let rl = left.rowids[l];
        let rr = right.rowids[r];
        debug_assert!(rl != rr);
        rowids[rowout] = if rl<rr {
            l+=1; interleave_left.insert(rowout); rl
        } else {
            r+=1; rr
        }
    }

    /* Sort out the rows */
    let (lmerge,lkeep) = left.contents.split_at_row(nmerge);
    left.contents.clear();
    let (rmerge,rkeep) = right.contents.split_at_row(nmerge);
    right.contents.clear();

    /* Merge and solve */
    let mut merged = lmerge.append_columns(&rmerge);
    let solution = merged.systematic_form()?;
    let lproj = solution.project_out(&lkeep, true);
    let rproj = solution.project_out(&rkeep, false);
    let interleaved = lproj.interleave_rows(&rproj, &interleave_left);

    Some(BlockGroup { contents: interleaved, solution: solution, empty:false, expected_rows:0, rowids:rowids })
}

/**
 * Backward solve step.
 * Solve the matrix by substitution.
 */
fn backward_solve(center: &mut BlockGroup, left:&mut BlockGroup, right:&mut BlockGroup) {
    let contents = center.solution.rhs.mul(&center.contents);
    let contents = contents.interleave_rows(&center.contents, &center.solution.echelon);

    if right.empty {
        left.contents = contents;
        *center = BlockGroup::new(0,0,0,true);
    } else {
        (left.contents, right.contents) = contents.split_at_row(left.solution.rhs.cols_main);
        *center = BlockGroup::new(0,0,0,true);
    }
}

/**
 * Return the number of blocks needed for a certain number of (key,value) pairs.
 *
 * Internal but maybe informative.
 */
pub fn blocks_required(rows:usize) -> usize {
    let mut cols = rows + EXTRA_ROWS;
    if OVERPROVISION > 0 { cols += cols / OVERPROVISION as usize; }
    cols += 8*BLOCKSIZE-1;
    cols = max(cols, 16*BLOCKSIZE);
    cols / (8*BLOCKSIZE)
}

/** At which level will this row be resolved by the solver? */
fn resolution(row:&LfrRow) -> u8 {
    let delta = row.block_positions[0] ^ row.block_positions[1];
    (BlockIdx::BITS - 1 - delta.leading_zeros()) as u8
}

/**
 * Utility: either generate a fresh hash key, or derive one from an existing
 * key and index.
 */
pub fn choose_key(base_key: Option<HasherKey>, n:usize) -> HasherKey {
    match base_key {
        None => {
            let mut key = [0u8; 16];
            OsRng.fill_bytes(&mut key);
            key
        },
        Some(key) => {
            let mut hasher = LfrHasher::new_with_key(&key);
            WhyHashing::DerivingNewKey.hash(&mut hasher);
            n.hash(&mut hasher);
            let hash = hasher.finish128();

            let mut ret = [0u8; 16];
            ret[0..8] .copy_from_slice(&hash.h1.to_le_bytes());
            ret[8..16].copy_from_slice(&hash.h2.to_le_bytes());
            ret
        }
    }
}

/** Untyped core of a mapping object. */
struct MapCore {
    /** The number of bits stored per (key,value) pair. */
    pub bits_per_value: usize,
    nblocks: usize,
    blocks: Vec<LfrBlock>,
}

impl MapCore {
    /** Deterministically choose a seed for the given row's output */
    fn seed_row(hash_key: &HasherKey, naug: usize, row:usize) -> [u8; 8] {
        let mut h = LfrHasher::new_with_key(hash_key);
        WhyHashing::RandomizingSolution.hash(&mut h);
        row.hash(&mut h);
        let mut out_as_u64 = h.finish128().h1;
        if naug < 64 { out_as_u64 &= (1<<naug)-1; }
        out_as_u64.to_le_bytes()
    }

    /** Solve a hierarchical matrix constructed for the uniform map */
    fn build(mut blocks: Vec<BlockGroup>, nblocks:usize, hash_key: &HasherKey) -> Option<MapCore> {
        let mut halfstride = 1;
        while halfstride < nblocks {
            let mut pos = 2*halfstride;
            while pos < blocks.len() {
                let nmerge = blocks[pos].expected_rows;
                let (l,r) = blocks.split_at_mut(pos);
                blocks[pos] = forward_solve(&mut l[pos-halfstride], &mut r[halfstride], nmerge)?;
                pos += 4*halfstride;
            }
            halfstride *= 2;
        }

        /* Initialize solution at random */
        let naug = blocks[halfstride].solution.rhs.cols_aug;
        let final_nrows = blocks[halfstride].solution.rhs.cols_main;
        let mut seed = Matrix::new(final_nrows,0,naug);
        let empty = [];
        for row in 0..final_nrows {
            seed.mut_set_row_as_bytes(row, &empty, &MapCore::seed_row(hash_key, naug, row));
        }
        blocks[halfstride].contents = seed;

        /* Backward solve steps */
        for blk in 0..nblocks {
            /* Tell the backsolver how many bits the solutions have */
            blocks[2*blk+1].solution.rhs = Matrix::new(0,LfrBlock::BITS as usize,0);
        }
        halfstride /= 2;
        while halfstride > 0 {
            let mut pos = 2*halfstride;
            while pos < blocks.len() {
                let (l,r)  = blocks.split_at_mut(pos);
                let (c,rr) = r.split_at_mut(halfstride);
                backward_solve(&mut c[0], &mut l[pos-halfstride], &mut rr[0]);
                pos += 4*halfstride;
            }
            halfstride /= 2;
        }

        /* Serialize to core */
        let mut core = vec![0; naug*nblocks];
        for blki in 0..nblocks {
            let block = &blocks[2*blki+1];
            debug_assert_eq!(block.contents.rows, BLOCKSIZE*8);
            for aug in 0..naug {
                let mut word = 0;
                /* PERF: this could be faster, but do we care? */
                for bit in 0..LfrBlock::BITS {
                    word |= (block.contents.get_aug_bit(bit as usize,aug) as LfrBlock) << bit;
                }
                core[blki*naug+aug] = word;
            }
        }

        Some(MapCore{
            bits_per_value: naug,
            nblocks: nblocks,
            blocks: core
        })
    }

    /** Try once to build from a vector of hashed items. */
    fn build_from_vec(
        vec: Vec<LfrRow>,
        bits_per_value:usize,
        hash_key:&HasherKey
    ) -> Option<MapCore> {
        let nitems = vec.len();
        let mask = if bits_per_value == Response::BITS as usize { !0 }
            else { (1<<bits_per_value)-1 };
        let nblocks = blocks_required(nitems);
        let nblocks_po2 = next_power_of_2_ge(nblocks);
        let n_resolutions = (usize::BITS - (nblocks-1).leading_zeros() + 1) as usize;
        const BYTES_PER_VALUE : usize = (Response::BITS as usize) / 8;

        /* Count the number of vectors in each block with each resolution */
        let mut block_res : Vec<Vec<RowIdx>> = (0..nblocks).map(|_| vec![0;n_resolutions]).collect();
        for row in &vec {
            let res = resolution(&row);
            block_res[row.block_positions[0] as usize][res as usize] += 1;
            block_res[row.block_positions[1] as usize][res as usize] += 1;
        }


        /* Set expected rows */
        let mut block_expected_rows = vec![0;nblocks_po2];
        for b in 0..nblocks {
            for r in 0..n_resolutions-1 {
                let j = 1<<r;
                if (b&j) == 0 {
                    let idx = (b&!(j-1)) + j;
                    block_expected_rows[idx] += block_res[b][r] as usize;
                } 
            }
        }

        /* Cumulative sum */
        for b in 0..nblocks {
            let mut total=0;
            for r in 0..n_resolutions {
                let tmp = block_res[b][r];
                block_res[b][r] = total;
                total += tmp;
            }
        }

        /* Create the blocks */
        let mut blocks : Vec<BlockGroup> = (0..nblocks_po2*2).map(|i|
            if ((i&1) != 0) && i < 2*nblocks {
                let nrows_blk = block_res[i/2][n_resolutions-1];
                BlockGroup::new(nrows_blk as usize,BLOCKSIZE*8, bits_per_value, false)
            } else {
                let mut ret = BlockGroup::new(0,0,0,true);
                ret.expected_rows = block_expected_rows[i/2];
                ret
            }
        ).collect();

        let mut which_row = 0;
        for row in vec {
            for i in [0,1] {
                let blki = row.block_positions[i] as usize;
                let blk = &mut blocks[2*blki+1];
                let blkres = &mut block_res[blki];
                let res = resolution(&row);
                let rowid = blkres[res as usize];
                blkres[res as usize] += 1;
                let aug_bytes = if i==0 { [0u8; BYTES_PER_VALUE] }
                    else { (row.augmented & mask).to_le_bytes() };
                blk.contents.mut_set_row_as_bytes(rowid as usize,&row.block_key[i].to_le_bytes(), &aug_bytes);
                blk.rowids[rowid as usize] = (res,which_row);
            }
            which_row += 1;
        }

        /* Solve it */
        Self::build(blocks, nblocks, hash_key)
    }

    /** Query this map at a given row */
    fn query(&self, row:LfrRow) -> Response {
        let mut ret = row.augmented;
        if self.bits_per_value < Response::BITS as usize {
            ret &= (1<<self.bits_per_value) - 1;
        }
        let p0 = row.block_positions[0] as usize;
        let p1 = row.block_positions[1] as usize;
        let [k0,k1] = row.block_key;
        let naug = self.bits_per_value;
        for bit in 0..naug {
            let get = (self.blocks[p0*naug+bit] & k0) ^ (self.blocks[p1*naug+bit] & k1);
            ret ^= ((get.count_ones() & 1) as Response) << bit;
        }
        ret
    }
}

/**
 * Compressed static function of type `T -> u64`.
 *
 * These objects can be constructed from a mapping type or vector of
 * pairs.  They can then be queried with a given `T`; if that `T` was
 * included in the map during construction, then the corresponding value
 * will be returned.  Otherwise, an arbitrary value will be returned.
 */
pub struct Map<T> {
    /** The SipHash key used to hash inputs. */
    hash_key: HasherKey,

    /** Untyped map object, consulted after hashing. */
    core: MapCore,

    /** Phantom to hold the type of T */
    _phantom: PhantomData<T>
}

/**
 * Approximate sets.
 *
 * These are a possible replacement for Bloom filters in static contexts.
 * They store a compressed representation of a set of objects.  From that
 * representation you can query whether an object is in the set.
 * 
 * There is a small, adjustable false positive probability.  There are no
 * false negatives.  That is, if you query an object that really was in the
 * set, you will always get `true`.  If you query an ojbect not in the set,
 * you will usually get `false`, but not always.
 */
pub struct ApproxSet<T> {
    /** The SipHash key used to hash inputs. */
    hash_key: HasherKey,

    /** Untyped map object, consulted after hashing. */
    core: MapCore,

    /** Phantom to hold the type of T */
    _phantom: PhantomData<T>,
}

/**
 * Options to build a Map or ApproxSet.
 * 
 * Implements `Default`, so you can get reasonable options
 * with BuildOptions::default().
 */
#[derive(PartialEq,Eq,Debug,Ord,PartialOrd)]
pub struct BuildOptions{
    /**
     * The operation to build a map or approximate set
     * fails around 1%-10% of the time.  The builder will
     * automatically try up to this number of times.  It is
     * recommended to try at least 20 times for typical use
     * cases, so that the failure probability is negligible.
     *
     * Note that building will always fail if the keys
     * are not unique, even if the values are consistent.
     * For example, `[(1,2),(1,2)]` will always fail to build.
     * To avoid this, either deduplicate the items yourself,
     * or pass a `HashMap` or `BTreeMap` (or `HashSet` or `BTreeSet`)
     * to the builder.
     *
     * Default: 256.
     */
    pub max_tries : usize,

    /**
     * Out-parameter from build: on which try did the build
     * succeed?
     */
    pub try_num: usize,

    /** 
     * Optional hash key to make building deterministic.
     * If a key is given, then the actual key used will be
     * derived from that key and from the try count.
     * If omitted, a random key will be selected for each try.
     *
     * Default: `None`.
     */
    pub key_gen   : Option<HasherKey>, 

    /**
     * Override the number of bits to return per value.
     * If given, all values will be truncated to that many
     * bits.  If omitted, the bit length of the largest
     * input value will be used.
     * 
     * When building an ApproxSet, this determines the failure
     * probability (which is 2^-`bits_per_value`) and the storage
     * required by the ApproxSet (which is about `bits_per_value`
     * per element in the set).  For building an ApproxSet, this
     * defaults to 8.
     *
     * Default: `None`.
     */
    pub bits_per_value : Option<u8>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            key_gen : None,
            max_tries : 256,
            bits_per_value : None,
            try_num: 0
        }
    }
}

/** Next power of 2 >= x */
fn next_power_of_2_ge(x:usize) -> usize {
    if x==0 { return 1; }
    1 << (usize::BITS - (x-1).leading_zeros())
}

impl <T:Hash> Map<T> {
    /**
     * Build a uniform map.
     *
     * This function takes an iterable collection of items `(k,v)` and
     * constructs a compressed mapping.  If you query `k` on the compressed
     * mapping, `query` will return the corresponding `v`.  If you query any `k`
     * not included in the original list, the return value will be arbitrary.
     *
     * You can pass a `HashMap<T,u64>`, `BTreeMap<T,u64>` etc.  If you pass a
     * non-mapping type such as `Vec<(T,u64)>` then be careful: any duplicate
     * `T` entries will cause the build to fail, possibly after a long time,
     * even if they have the same u64 associated.
     */
    pub fn build<'a, Collection>(map: &'a Collection, options: &mut BuildOptions) -> Option<Self>
    where for<'b> &'b Collection: IntoIterator<Item=(&'b T, &'b Response)>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Get the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => {
                let mut range : Response = 0;
                for v in map.into_iter().map(|(_k,v)| v) { range |= v; }
                (Response::BITS - range.leading_zeros()) as usize
            },
            Some(bpv) => bpv as usize
        };

        for try_num in 0..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let iter1 = map.into_iter();
            let nblocks = blocks_required(iter1.len());
            let vec = iter1.map(|(k,v)| {
                let mut row = hash_object_to_row(&hkey,nblocks,k);
                row.augmented ^= v;
                row
            }).collect();

            /* Solve it! (with type-independent code) */
            if let Some(solution) = MapCore::build_from_vec(vec, bits_per_value, &hkey) {
                options.try_num = try_num;
                return Some(Self {
                    hash_key: hkey,
                    core: solution,
                    _phantom: PhantomData::default()
                });
            }
        }

        None // Fail!
    }

    /**
     * Query an item in the map.
     * If (key,v) was included when building the map, then v will be returned.
     * Otherwise, an arbitrary value will be returned.
     */
    pub fn query(&self, key: &T) -> u64 {
        self.core.query(hash_object_to_row(&self.hash_key, self.core.nblocks, key))
    }
}

impl <T:Hash> ApproxSet<T> {
    /** Default bits per value if none is specified. */
    const DEFAULT_BITS_PER_VALUE : usize = 8;

    /**
     * Build an approximate set.
     *
     * This function takes an iterable collection of items `x` and constructs
     * a compressed representation of the collection.  If you query `k` on the compressed
     * mapping, `query` will return the corresponding `v`.  If you query any `k`
     * not included in the original list, the return value will be arbitrary.
     *
     * You can pass a `HashSet<T>`, `BTreeSet<T>` etc.  If you pass a non-set
     * type such as `Vec<T>` then be careful: any duplicate `T` entries will
     * cause the build to fail, possibly after a long time.
     */
    pub fn build<'a, Collection>(set: &'a Collection, options: &mut BuildOptions) -> Option<Self>
    where for<'b> &'b Collection: IntoIterator<Item=&'b T>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Choose the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => Self::DEFAULT_BITS_PER_VALUE,
            Some(bpv) => bpv as usize
        };

        for try_num in 0..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let iter1 = set.into_iter();
            let nblocks = blocks_required(iter1.len());
            let vec = iter1.map(|k| hash_object_to_row(&hkey,nblocks,k)).collect();

            /* Solve it! (with type-independent code) */
            if let Some(solution) = MapCore::build_from_vec(vec, bits_per_value, &hkey) {
                options.try_num = try_num;
                return Some(Self {
                    hash_key: hkey,
                    core: solution,
                    _phantom: PhantomData::default()
                });
            }
        }

        None // Fail!
    }

    /**
     * Query whether an item is in the set.
     *
     * If `key` was included in the iterator when building the
     * `ApproxSet`, then return `true`.  Otherwise, usually return `false`, but there is
     * a small false positive rate depending on set construction parameters.  Queries,
     * and thus false positives, are deterministic after the set has been constructed.
     */
    pub fn probably_contains(&self, key: &T) -> bool {
        self.core.query(hash_object_to_row(&self.hash_key, self.core.nblocks, key)) == 0
    }
}

#[cfg(test)]
mod tests {
    use crate::uniform::{ApproxSet,Map,BuildOptions};
    use rand::{thread_rng,Rng};
    use std::collections::{HashMap,HashSet};

    /* Test map functionality */
    #[test]
    fn test_uniform_map() {
        let mut rng = thread_rng();
        let mut map = HashMap::new();
        for i in 0..10 {
            for _j in 0..99*i {
                map.insert(rng.gen::<u64>(), rng.gen::<u64>());
            }
            let hiermap = Map::build(&map, &mut BuildOptions::default()).unwrap();
            for (k,v) in map.iter() {
                assert_eq!(hiermap.query(&k), *v);
            }
        }
    }

    /* Test set functionality */
    #[test]
    fn test_approx_set() {
        let mut rng = thread_rng();
        let mut set = HashSet::new();
        for i in 0..10 {
            for _j in 0..99*i {
                set.insert(rng.gen::<u64>());
            }
            
            let hiermap = ApproxSet::build(&set, &mut BuildOptions::default()).unwrap();
            for k in set.iter() {
                assert_eq!(hiermap.probably_contains(&k), true);
            }

            let mut false_positives = 0;
            let ntries = 10000;
            let mu = ntries as f64 / 2f64.powf(hiermap.core.bits_per_value as f64);
            for _ in 0..ntries {
                false_positives += hiermap.probably_contains(&rng.gen::<u64>()) as usize;
            }
            // 4 standard deviations
            assert!((false_positives as f64) < mu + 4.*mu.sqrt());
        }
    }
}