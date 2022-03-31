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
use std::mem::{drop,replace};
use siphasher::sip128::{Hasher128, SipHasher13, Hash128};
use serde::{Serialize,Deserialize};
use std::sync::{Arc,Mutex,Condvar};
use std::thread::{spawn,available_parallelism,JoinHandle};
use std::sync::atomic::{AtomicBool,AtomicUsize,Ordering};
use rayon::prelude::*;

/** Space of responses in dictionary impl. */
pub type Response = u64;

type LfrHasher = SipHasher13;

/** A key for the SipHash13 hash function. */
pub type HasherKey = [u8; 16];

type RowIdx   = usize; // if u32 then limit to ~4b rows
type BlockIdx = u32;
pub(crate) type LfrBlock = u32;

/** The internal block size of the map, in bytes. */
pub const BLOCKSIZE : usize = (LfrBlock::BITS as usize) / 8;
const OVERPROVISION : u64 = /* 1/ */ 1024;
const EXTRA_ROWS : usize = 8;

/** Result of hashing an item */
#[derive(Copy,Clone,Eq,PartialEq,Ord,PartialOrd,Debug)]
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
    let num = smoothlog * leading_coefficient; // With BLOCKSIZE=4, can'K overflow until k=85 which is impossible

    stride_seed |= (1u64<<33) / OVERPROVISION; // | instead of + because it can'K overflow
    let den = (((stride_seed * stride_seed) as u128 * nblocks_huge) >> 32) as u64; // can'K overflow because stride_seed is only 32 bits
    let b_seed = (num / den + a_seed) & 0xFFFFFFFF; // den can'K be 0 because stride_seed is adjusted

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
fn hash_object_to_row<K:Hash> (key: &HasherKey, nblocks:usize, k: &K)-> LfrRow {
    let mut h = LfrHasher::new_with_key(&key);
    WhyHashing::HashingInput.hash(&mut h);
    k.hash(&mut h);
    interpret_hash_as_row(h.finish128(), nblocks)
}

enum ChannelResult<T> {
    Empty,
    Full(T),
    Expired
}
use ChannelResult::*;

/**
 * Single-use channel for communicating intermediate results.
 * 
 * Each channel may be used only once.
 * The operation order is new() -> 
 */
struct Channel<T> {
    current: Mutex<ChannelResult<T>>,
    condvar: Condvar,
    available: AtomicBool
}

impl<T> Channel<T> {
    fn new() -> Self {
        Channel {
            current: Mutex::new(Empty),
            condvar: Condvar::new(),
            available: AtomicBool::new(true)
        }
    }
    
    fn ctake(&self) -> Option<T> {
        let mut cur = self.current.lock().ok()?;
        while let Empty = &*cur { cur = self.condvar.wait(cur).ok()? };
        match replace(&mut *cur, Expired) {
            Empty => None, // can't happen but whatev
            Expired => None,
            Full(x) => Some(x)
        }
    }

    fn write(&self, t:T) -> Option<()> {
        let mut cur = self.current.lock().ok()?;
        *cur = Full(t);
        self.condvar.notify_all();
        Some(())
    }

    fn expire(&self) -> Option<()> {
        let mut cur = self.current.lock().ok()?;
        *cur = Expired;
        self.condvar.notify_all();
        Some(())
    }

    fn reserve(&self) -> bool {
        self.available.swap(false,Ordering::Relaxed)
    }
}

/** A block or group of blocks for the hierarchical solver */
struct BlockGroup {
    /* Forward solution; row IDs in solution; number of expected rows in reverse solution */
    fwd_solution:   Channel<(Matrix, Vec<RowIdx>, usize)>, // with RowIdx::MAX appended for convenience
    bwd_solution:   Channel<Matrix>,
    /* Systematic form of solution; number of expected rows in reverse solution for left side */
    systematic:     Channel<(Systematic,usize)>
}

impl BlockGroup {
    fn new() -> BlockGroup {
        BlockGroup {
            bwd_solution: Channel::new(),
            fwd_solution: Channel::new(),
            systematic:   Channel::new()
        }
    }
}

/**
 * Forward solve step.
 * Merge the resolvable rows of the left and right BlockGroups, then
 * project them out of the remaining ones.  Clears the fwd_solution of left and right,
 * but not the solution.
 *
 * Fails if the block group doesn'K have full row-rank.
 */
fn forward_solve(left:&BlockGroup, right:&BlockGroup, center: &BlockGroup) -> Option<()> {
    if !center.fwd_solution.reserve() { return Some(()); }
    let (lsol, lrow_ids, lload) = if let Some(x) = left.fwd_solution.ctake() { x } else {
        center.systematic.expire();
        center.fwd_solution.expire();
        return None;
    };
    let (rsol, rrow_ids, _rload) = if let Some(x) = right.fwd_solution.ctake() { x } else {
        center.systematic.expire();
        center.fwd_solution.expire();
        return None;
    };

    if rsol.rows == 0 && rsol.cols_main == 0 {
        /* empty */
        center.fwd_solution.write((lsol, lrow_ids, lload));
        center.systematic.write((Systematic::identity(0),lload));
        return Some(());
    }

    /* Mergesort the row IDs */
    let mut left_to_merge = BitSet::with_capacity(lsol.rows);
    let mut right_to_merge = BitSet::with_capacity(rsol.rows);
    let mut interleave_left = BitSet::with_capacity(lsol.rows+rsol.rows);
    let mut row_ids = Vec::new();
    row_ids.reserve(lsol.rows + rsol.rows); // over-reserve but this isn'K the long pole
    let (mut l, mut r, mut i) = (0,0,0);

    /* There should always be something in the left and right group,
        * because we appended RowIdx::MAX
        */
    while (l < lsol.rows) || (r < rsol.rows) {
        if lrow_ids[l] == rrow_ids[r] {
            left_to_merge.insert(l);
            right_to_merge.insert(r);
            l += 1;
            r += 1;
        } else if lrow_ids[l] < rrow_ids[r] {
            row_ids.push(lrow_ids[l]);
            interleave_left.insert(i);
            l += 1;
            i += 1;
        } else {
            row_ids.push(rrow_ids[r]);
            r += 1;
            i += 1;
        }
    }
    row_ids.push(RowIdx::MAX);

    /* Sort out the rows */
    let (lmerge,lkeep) = lsol.partition_rows(&left_to_merge);
    let (rmerge,rkeep) = rsol.partition_rows(&right_to_merge);

    /* Merge and solve */
    let mut merged = lmerge.append_columns(&rmerge);
    if let Some(systematic) = merged.systematic_form() {
        let lproj = systematic.project_out(&lkeep, true);
        let rproj = systematic.project_out(&rkeep, false);
        let interleaved = lproj.interleave_rows(&rproj, &interleave_left);
        center.fwd_solution.write((interleaved, row_ids, systematic.rhs.cols_main));
        center.systematic.write((systematic,lload));
        Some(())
    } else {
        center.systematic.expire();
        center.fwd_solution.expire();
        None
    }
}

/**
 * Backward solve step.
 * Solve the matrix by substitution.
 */
fn backward_solve(center: &BlockGroup, left:&BlockGroup, right:&BlockGroup) -> Option<()> {
    if !left.bwd_solution.reserve() { return Some(()); }
    let csol = if let Some(csol) = center.bwd_solution.ctake() {
        csol
    } else {
        left.bwd_solution.expire();
        right.bwd_solution.expire();
        return None;
    };
    let (csys,load) = if let Some(x) = center.systematic.ctake() { x } else {
        left.bwd_solution.expire();
        right.bwd_solution.expire();
        return None;
    };
    let solution = csys.rhs.mul(&csol);
    let solution = solution.interleave_rows(&csol, &csys.echelon);
    let (lsol,rsol) = solution.split_at_row(load);
    left.bwd_solution.write(lsol);
    right.bwd_solution.write(rsol);
    Some(())
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

/**
 * Utility: either generate a fresh hash key, or derive one from an existing
 * key and index.
 */
pub(crate) fn choose_key(base_key: Option<HasherKey>, n:usize) -> HasherKey {
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

/** Form of MapCore with validation and smaller serialization */
#[derive(Serialize,Deserialize)]
struct MapCoreSer {
    hash_key: HasherKey,
    bits_per_value: u8,
    blocks: Vec<LfrBlock>,
}

/** Untyped core of a mapping object. */
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Debug,Serialize,Deserialize)]
#[serde(try_from="MapCoreSer",into="MapCoreSer")]
pub(crate) struct MapCore {
    /** The SipHash key used to hash inputs. */
    pub(crate) hash_key: HasherKey,

    /** The number of bits stored per (key,value) pair. */
    pub bits_per_value: u8,

    /** The number of blocks per bit */
    pub nblocks: usize,

    /** The block data itself */
    pub(crate) blocks: Vec<LfrBlock>,
}

impl TryFrom<MapCoreSer> for MapCore {
    type Error = &'static str;
    fn try_from(ser: MapCoreSer) -> Result<MapCore,&'static str> {
        let nblocks = if ser.bits_per_value == 0 {
            if ser.blocks.len() != 0 { return Err("can't have blocks if bits_per_value==0"); }
            2
        } else if ser.bits_per_value > Response::BITS as u8 {
            return Err("can't have have more than Response::BITS per value");
        } else if ser.blocks.len() % ser.bits_per_value as usize != 0 {
            return Err("bits_per_value must evenly divide blocks.len()");
        } else {
            ser.blocks.len() / ser.bits_per_value as usize
        };
        if nblocks < 2 {
            Err("must have nblocks >= 2")
        } else {
            Ok(MapCore{
                hash_key: ser.hash_key,
                bits_per_value: ser.bits_per_value,
                blocks: ser.blocks,
                nblocks: nblocks
            })
        }
    }
}

impl From<MapCore> for MapCoreSer {
    fn from(core:MapCore) -> MapCoreSer {
        MapCoreSer {
            hash_key: core.hash_key,
            bits_per_value: core.bits_per_value,
            blocks: core.blocks
        }
    } 
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
    fn build(blocks: &Vec<BlockGroup>, naug:usize, nblocks:usize, hash_key: HasherKey) -> Option<()> {
        let mut halfstride = 1;
        while halfstride < nblocks {
            let mut pos = 2*halfstride;
            while pos < blocks.len() {
                forward_solve(&blocks[pos-halfstride], &blocks[pos+halfstride], &blocks[pos])?;
                pos += 4*halfstride;
            }
            halfstride *= 2;
        }

        /* Initialize backward solution at random */
        if blocks[halfstride].bwd_solution.reserve() {
            if let Some((_,_,load)) = blocks[halfstride].fwd_solution.ctake() {
                let mut seed = Matrix::new(0,0,naug);
                seed.reserve_rows(load);
                let empty = [];
                for row in 0..load {
                    seed.mut_add_row_as_bytes(&empty, &MapCore::seed_row(&hash_key, naug, row));
                }
                blocks[halfstride].bwd_solution.write(seed);
            } else {
                blocks[halfstride].bwd_solution.expire();
            }
        }

        /* Backward solve steps */
        halfstride /= 2;
        while halfstride > 0 {
            let mut pos = 2*halfstride;
            while pos < blocks.len() {
                backward_solve(&blocks[pos], &blocks[pos-halfstride], &blocks[pos+halfstride])?;
                pos += 4*halfstride;
            }
            halfstride /= 2;
        }

        Some(())
    }

    /** Try once to build from an iterator. */
    fn build_from_iter<'a,'b:'a, K:'b+Hash+Sync, V:'b+Into<Response>+Copy, Iter:ExactSizeIterator<Item=(&'b K,&'b V)>>(
        iter: &'a mut Iter,
        bits_per_value:u8,
        hash_key:&HasherKey,
        shift: u8
    ) -> Option<MapCore> {
        if bits_per_value == 0 {
            /* force nblocks to 2 */
            return Some(MapCore{
                hash_key: *hash_key,
                bits_per_value: bits_per_value,
                nblocks: 2,
                blocks: vec![],
            });
        }

        let nitems = iter.len();
        let mask = if bits_per_value == Response::BITS as u8 { !0 }
            else { (1<<bits_per_value)-1 };
        let nblocks = blocks_required(nitems);
        let nblocks_po2 = next_power_of_2_ge(nblocks);
        const BYTES_PER_VALUE : usize = (Response::BITS as usize) / 8;

        /* Create the blocks */
        let blocks : Vec<Mutex<(Matrix,Vec<RowIdx>)>> = (0..nblocks).map(|_| {
            let mut blk = Matrix::new(0,BLOCKSIZE*8, bits_per_value as usize);
            blk.reserve_rows(BLOCKSIZE*8*2 * 5/4);
            let rowids = Vec::with_capacity(BLOCKSIZE*8*2 * 5/4);
            Mutex::new((blk,rowids))
        }).collect();


        /* Seed the items into the blocks.
         * Perf: need to parallelize the hashing!
         */
        let items_vec: Vec<(&'a K, Response)> = iter.map(|(k,v)| (k,(*v).into())).collect();
        let parallelism = available_parallelism().map(|x| x.get()).unwrap_or(1);
        let atomic_rowi = AtomicUsize::new(0);
        (0..items_vec.len()).into_par_iter().for_each(|i| {
            let (k,v) = items_vec[i];
            let mut row = hash_object_to_row(hash_key, nblocks, k);
            row.augmented ^= v >> shift;

            /* Acquire mutexes in order */
            let (mut g0, mut g1) = if row.block_positions[0] < row.block_positions[1] {
                let g0 = blocks[row.block_positions[0] as usize].lock().unwrap();
                let g1 = blocks[row.block_positions[1] as usize].lock().unwrap();
                (g0,g1)
            } else {
                let g1 = blocks[row.block_positions[1] as usize].lock().unwrap();
                let g0 = blocks[row.block_positions[0] as usize].lock().unwrap();
                (g0,g1)
            };
            let rowi = atomic_rowi.fetch_add(1,Ordering::Relaxed);
            let (ref mut b0, ref mut r0) = *g0;
            let (ref mut b1, ref mut r1) = *g1;
            b0.mut_add_row_as_bytes(&row.block_key[0].to_le_bytes(), &[0u8; BYTES_PER_VALUE]);
            b1.mut_add_row_as_bytes(&row.block_key[1].to_le_bytes(), &(row.augmented & mask).to_le_bytes());
            r0.push(rowi);
            r1.push(rowi);
        });
        drop(items_vec);
        
        /* Create the block groups from the blocks */
        let block_groups : Vec<BlockGroup> = (0..nblocks_po2*2).map(|_| BlockGroup::new()).collect();

        for (i,blk) in blocks.into_iter().enumerate() {
            let (ref mut blk, ref mut row_ids) = *blk.lock().unwrap();
            let blk = replace(blk,Matrix::new(0,0,0));
            let mut row_ids = replace(row_ids, Vec::new());
            row_ids.push(RowIdx::MAX); /* Append guard rowidx */
            block_groups[2*i+1].fwd_solution.write((blk,row_ids,LfrBlock::BITS as usize));
        }
        for i in nblocks..nblocks_po2 {
            block_groups[2*i+1].fwd_solution.write((Matrix::new(0,0,0),Vec::new(),0));
        }

        /* Solve it */
        let naug = bits_per_value as usize;
        let hash_key_lit = *hash_key;
        let arc_block_groups = Arc::new(block_groups);
        let joins : Vec<JoinHandle<Option<()>>> = (0..parallelism-1).map(|_| {
            let arc_block_groups = Arc::clone(&arc_block_groups);
            spawn(move || Self::build(arc_block_groups.as_ref(), naug, nblocks, hash_key_lit))
        }).collect();
        let result = Self::build(arc_block_groups.as_ref(), naug, nblocks, hash_key_lit);
        for j in joins {
            j.join().ok()??;
        }
        result?;

        /* Serialize to core */
        let mut core = vec![0; naug*nblocks];
        for blki in 0..nblocks {
            let bsol = arc_block_groups[2*blki+1].bwd_solution.ctake().unwrap();
            debug_assert_eq!(bsol.rows, BLOCKSIZE*8);
            for aug in 0..naug {
                let mut word = 0;
                /* PERF: this could be faster, but do we care? */
                for bit in 0..LfrBlock::BITS {
                    word |= (bsol.get_aug_bit(bit as usize,aug) as LfrBlock) << bit;
                }
                core[blki*naug+aug] = word;
            }
        }

        Some(MapCore{
            bits_per_value: bits_per_value,
            blocks: core,
            nblocks: nblocks,
            hash_key: *hash_key
        })
    }

    /** Query this map at a given row */
    fn query(&self, row:LfrRow) -> Response {
        let mut ret = row.augmented;
        if self.bits_per_value < Response::BITS as u8 {
            ret &= (1<<self.bits_per_value) - 1;
        }
        let p0 = row.block_positions[0] as usize;
        let p1 = row.block_positions[1] as usize;
        let [k0,k1] = row.block_key;
        let naug = self.bits_per_value as usize;
        for bit in 0..naug {
            let get = (self.blocks[p0*naug+bit] & k0) ^ (self.blocks[p1*naug+bit] & k1);
            ret ^= ((get.count_ones() & 1) as Response) << bit;
        }
        ret
    }

    /** Query this map at the hash of a given key */
    pub(crate) fn query_hash<K:Hash>(&self, key: &K) -> Response {
        self.query(hash_object_to_row(&self.hash_key, self.nblocks, key))
    }
}

/**
 * Lower-level: compressed uniform static functions.
 * 
 * These functions are a building block of the generic case.  They are
 * efficient when the value being mapped to is approximately uniformly
 * random from a power-of-2 interval, i.e. all values are about equally
 * likely.
 * 
 * They don't store a table of `V`'s: instead they are limited to
 * returning numeric types of at most 64 bits.
 */
#[derive(Eq,PartialEq,Ord,PartialOrd,Debug,Serialize,Deserialize)]
#[serde(try_from="MapCoreSer",into="MapCoreSer")]
pub struct CompressedRandomMap<K,V> {
    /** Untyped map object, consulted after hashing. */
    pub(crate) core: MapCore,

    /** Phantom to hold the types of K,V */
    _phantom: PhantomData<(K,V)>
}

impl <K,V> Clone for CompressedRandomMap<K,V> {
    fn clone(&self) -> Self {
        CompressedRandomMap { core: self.core.clone(), _phantom:PhantomData::default() }
    }
}

impl <K,V> TryFrom<MapCoreSer> for CompressedRandomMap<K,V> {
    type Error = &'static str;
    fn try_from(ser: MapCoreSer) -> Result<CompressedRandomMap<K,V>,&'static str> {
        let core = MapCore::try_from(ser)?;
        Ok(CompressedRandomMap { core: core, _phantom: PhantomData::default() })
    }
}

impl <K,V> From<CompressedRandomMap<K,V>> for MapCoreSer {
    fn from(map: CompressedRandomMap<K,V>) -> MapCoreSer { MapCoreSer::from(map.core) }
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
 * set, you will always get `true`.  If you query an object not in the set,
 * you will usually get `false`, but not always.
 */
#[derive(Eq,PartialEq,Ord,PartialOrd,Debug,Serialize,Deserialize)]
#[serde(try_from="MapCoreSer",into="MapCoreSer")]
pub struct ApproxSet<K> {
    /** Untyped map object, consulted after hashing. */
    core: MapCore,

    /** Phantom to hold the type of K */
    _phantom: PhantomData<K>,
}

impl <K> Clone for ApproxSet<K> {
    fn clone(&self) -> Self {
        ApproxSet { core: self.core.clone(), _phantom:PhantomData::default() }
    }
}

impl <K> TryFrom<MapCoreSer> for ApproxSet<K> {
    type Error = &'static str;
    fn try_from(ser: MapCoreSer) -> Result<ApproxSet<K>,&'static str> {
        let core = MapCore::try_from(ser)?;
        Ok(ApproxSet { core: core, _phantom: PhantomData::default() })
    }
}

impl <K> From<ApproxSet<K>> for MapCoreSer {
    fn from(set: ApproxSet<K>) -> MapCoreSer { MapCoreSer::from(set.core) }
}

/**
 * Options to build a [`CompressedMap`](crate::CompressedMap),
 * [`CompressedRandomMap`] or [`ApproxSet`].
 * 
 * Implements `Default`, so you can get reasonable options
 * with `BuildOptions::default()`.
 */
#[derive(Copy,Clone,PartialEq,Eq,Debug,Ord,PartialOrd)]
pub struct BuildOptions{
    /**
     * How many times to try building the set?
     * 
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
     * In-out-parameter from build.
     * 
     * On which try did the build succeed?  If passed in
     * as nonzero, the counter starts here.  Mostly useful
     * for diagnostics.
     */
    pub try_num: usize,

    /** 
     * Optional hash key to make building deterministic.
     * If a key is given, then the actual key used will be
     * derived from that key and from `try_num`.
     * If omitted, a fresh random key will be selected for
     * each try.
     *
     * Default: `None`.
     */
    pub key_gen : Option<HasherKey>, 

    /**
     * Override the number of bits to return per value.
     * If given, all values will be truncated to that many
     * least-significant bits.  If omitted, the bit length
     * of the largest input value will be used.
     * 
     * When building an [`ApproxSet`], this determines the failure
     * probability (which is 2<sup>-`bits_per_value`</sup>) and
     * the storage required by the ApproxSet (which is about
     * `bits_per_value` per element in the set).
     * 
     * Ignored by [`CompressedMap`](crate::CompressedMap).
     *
     * Default: `None`.  When building an [`ApproxSet`], `None`
     * will be interpreted as 8 bits per value.
     */
    pub bits_per_value : Option<u8>,

    /**
     * When constructing a [`CompressedRandomMap`], shift the
     * inputs right by this amount.  This is used by
     * [`CompressedMap`](crate::CompressedMap) to construct
     * several [`CompressedRandomMap`]s capturing different
     * bits of the input, without rewriting a giant vector.
     * 
     * TODO: maybe remove this and handle a different way?
     * 
     * Default: 0.
     */
    pub shift: u8
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            key_gen : None,
            max_tries : 256,
            bits_per_value : None,
            try_num: 0,
            shift: 0
        }
    }
}

/** Next power of 2 >= x */
fn next_power_of_2_ge(x:usize) -> usize {
    if x==0 { return 1; }
    1 << (usize::BITS - (x-1).leading_zeros())
}

impl <K:Hash+Sync,V:Copy> CompressedRandomMap<K,V>
where Response:From<V> {
    /**
     * Build a uniform map.
     *
     * This function takes an iterable collection of items `(k,v)` and
     * constructs a compressed mapping.  If you query `k` on the compressed
     * mapping, `query` will return the corresponding `v`.  If you query any `k`
     * not included in the original list, the return value will be arbitrary.
     *
     * You can pass a `HashMap<K,V>`, `BTreeMap<K,V>` etc.  If you pass a
     * non-mapping type such as `Vec<(K,V)>` then be careful: any duplicate
     * `K` entries will cause the build to fail, possibly after a long time,
     * even if they have the same u64 associated.
     */
    pub fn build<'a, Collection>(map: &'a Collection, options: &mut BuildOptions) -> Option<Self>
    where K:'a, V:'a,
          for<'b> &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator,
    {
        /* Get the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => {
                let mut range : Response = 0;
                for v in map.into_iter().map(|(_k,v)| v) { range |= Response::from(*v); }
                (Response::BITS - range.leading_zeros()) as u8
            },
            Some(bpv) => bpv as u8
        };

        for try_num in options.try_num..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let mut iter = map.into_iter();
            if let Some(solution) = MapCore::build_from_iter(&mut iter, bits_per_value, &hkey, options.shift) {
                options.try_num = try_num;
                return Some(Self {
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
    pub fn query(&self, key: &K) -> V 
    where V:From<Response> {
        self.core.query_hash(key).into()
    }

    /**
     * Query an item in the map.
     * If (key,v) was included when building the map, then v will be returned.
     * Otherwise, an arbitrary value will be returned.
     */
    pub fn try_query(&self, key: &K) -> Option<V> 
    where V:TryFrom<Response> {
        self.core.query_hash(key).try_into().ok()
    }
}

impl <K:Hash+Sync> ApproxSet<K> {
    /** Default bits per value if none is specified. */
    const DEFAULT_BITS_PER_VALUE : u8 = 8;

    /**
     * Build an approximate set.
     *
     * This function takes an iterable collection `set` of items and constructs
     * a compressed representation of the collection.  If you query
     * `result.probably_contains(x)`, then if `set.contains(x)` you will
     * always get `true`.  If `!set.contains(x)` you will usually get `false`,
     * but with some small probability you will instead get `true`.  The false
     * positive probability can be configured by changing the `bits_per_value`
     * of the [`BuildOptions`] you pass: it is approximately 2<sup>-`bits_per_value`</sup>,
     * and the storage required for the set is approximately `bits_per_value` bits
     * per element of `set`.
     *
     * You can pass a `HashSet<K>`, `BTreeSet<K>` etc.  If you pass a non-set
     * type then be careful: any duplicate `K` entries will cause the build
     * to fail, possibly after a long time.
     */
    pub fn build<'a, Collection>(set: &'a Collection, options: &mut BuildOptions) -> Option<Self>
    where for<'b> &'b Collection: IntoIterator<Item=&'b K>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Choose the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => Self::DEFAULT_BITS_PER_VALUE,
            Some(bpv) => bpv as u8
        };

        for try_num in options.try_num..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let iter1 = set.into_iter();
            let zero = 0u64;
            let mut iter = iter1.map(|k| (k,&zero) );
            if let Some(solution) = MapCore::build_from_iter(&mut iter, bits_per_value, &hkey, 0) {
                options.try_num = try_num;
                return Some(Self {
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
    pub fn probably_contains(&self, key: &K) -> bool {
        self.core.query_hash(key) == 0
    }
}

#[cfg(test)]
mod tests {
    use crate::uniform::{ApproxSet,CompressedRandomMap,BuildOptions};
    use rand::{thread_rng,Rng,SeedableRng};
    use rand::rngs::StdRng;
    use std::collections::{HashMap,HashSet};
    // use serde::{Serialize,Deserialize};

    /* Test map functionality */
    #[test]
    fn test_uniform_map() {
        let mut map = HashMap::new();
        for i in 0u32..100 {
            let mut seed = [0u8;32];
            let mut rng : StdRng = SeedableRng::from_seed(seed);
            seed[0..4].copy_from_slice(&i.to_le_bytes());
            for _j in 0..99*((i+9)/10) {
                map.insert(rng.gen::<u64>(), rng.gen::<u8>());
            }
            let mut options = BuildOptions::default();
            options.key_gen = Some(seed[..16].try_into().unwrap());
            let hiermap = CompressedRandomMap::<u64,u8>::build(&map, &mut options).unwrap();

            for (k,v) in map.iter() {
                assert_eq!(hiermap.try_query(&k), Some(*v));
            }

            // let ser = hiermap.serialize();
            // let deser = CompressedRandomMap::<u64,u8>::deserialize(&ser);
            // assert!(deser.is_some());
            // assert_eq!(hiermap, deser.unwrap()); 
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
            // println!("{} false positives", false_positives);
            assert!((false_positives as f64) < mu + 4.*mu.sqrt());

            // let ser = hiermap.serialize();
            // let deser = ApproxSet::<u64>::deserialize(&ser);
            // assert!(deser.is_some());
            // assert_eq!(hiermap, deser.unwrap()); 
        }
    }
}