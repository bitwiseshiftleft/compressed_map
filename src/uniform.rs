/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Uniform sparse linear map implementation.
 */

use crate::tilematrix::matrix::{Matrix,Systematic};
use crate::tilematrix::bitset::BitSet;
use core::marker::PhantomData;
use core::hash::Hash;
use core::cmp::max;
use rand::{RngCore};
use rand::rngs::OsRng;
use siphasher::sip128::{Hasher128, SipHasher13, Hash128};

use bincode::{Encode,Decode};
use bincode::enc::{Encoder};
use bincode::de::{Decoder};
use bincode::error::{EncodeError,DecodeError};
use bincode::enc::write::Writer;
use bincode::de::read::Reader;

#[cfg(feature="threading")]
use {
    std::sync::{Arc,Mutex,Condvar},
    std::thread::{spawn,available_parallelism,JoinHandle},
    std::sync::atomic::{AtomicBool,Ordering},
    std::mem::replace
};

#[cfg(not(feature="threading"))]
use core::cell::RefCell;

use std::borrow::Cow;

/** Space of responses in dictionary impl. */
pub type Response = u64;

type LfrHasher = SipHasher13;

/** A key for the SipHash13 hash function. */
pub type HasherKey = [u8; 16];

type RowIdx   = u64; // if u32 then limit to ~4b rows
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

#[allow(dead_code)]
enum ChannelResult<T> {
    Empty,
    Full(T),
    Expired
}
#[allow(unused_imports)]
use ChannelResult::*;

/**
 * Single-use channel for communicating intermediate results.
 * 
 * Each channel may be used only once.
 * The operation order is new() -> 
 */
#[cfg(feature="threading")]
struct Channel<T> {
    current: Mutex<ChannelResult<T>>,
    condvar: Condvar,
    available: AtomicBool
}

#[cfg(feature="threading")]
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

/** Unthreaded version of a channel is much simpler */
#[cfg(not(feature="threading"))]
struct Channel<T> {
    current: RefCell<Option<T>>
}

#[cfg(not(feature="threading"))]
impl<T> Channel<T> {
    fn new() -> Self { Channel { current: RefCell::new(None) }}
    fn ctake(&self) -> Option<T> { self.current.replace(None) }
    fn write(&self, t:T) -> Option<()> { *self.current.borrow_mut() = Some(t); Some(()) }
    fn expire(&self) -> Option<()> { *self.current.borrow_mut() = None; Some(()) }
    fn reserve(&self) -> bool { true }
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

/** Untyped core of a mapping object. */
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Debug)]
pub(crate) struct MapCore<'a> {
    /** The SipHash key used to hash inputs. */
    pub(crate) hash_key: HasherKey,

    /** The number of bits stored per (key,value) pair. */
    pub bits_per_value: u8,

    /** The number of blocks per bit */
    pub nblocks: usize,

    /** The block data itself */
    pub(crate) blocks: Cow<'a, [u8]>,
}

impl <'a> Encode for MapCore<'a> {
    fn encode<'b,E: Encoder>(&'b self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.hash_key, encoder)?;
        Encode::encode(&self.bits_per_value, encoder)?;
        Encode::encode(&self.nblocks, encoder)?;
        encoder.writer().write(&self.blocks.as_ref())
    }
}

impl <'a> Decode for MapCore<'a> {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let hash_key       = Decode::decode(decoder)?;
        let bits_per_value = Decode::decode(decoder)?;
        let nblocks : usize = Decode::decode(decoder)?;
        if nblocks < 2 {
            return Err(DecodeError::OtherString("bits_per_value must be at least 2".to_string()));
        } else if bits_per_value == 0 && nblocks != 2 {
            return Err(DecodeError::OtherString("bits_per_value must be exactly 2 for trivial map".to_string()));
        }
        let mul1 : usize = nblocks.checked_mul(BLOCKSIZE)
            .ok_or(DecodeError::OtherString("overflow on multiply".to_string()))?;
        let mul2 : usize = mul1.checked_mul(bits_per_value as usize)
            .ok_or(DecodeError::OtherString("overflow on multiply".to_string()))?;
        let mut blocks = vec![0u8;mul2];
        decoder.reader().read(&mut blocks)?;
        Ok(MapCore {
            hash_key: hash_key,
            bits_per_value: bits_per_value,
            blocks: Cow::Owned(blocks),
            nblocks: nblocks
        })
    }
}

impl <'a> MapCore<'a> {
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
    fn build_from_iter(
        iter: &mut dyn ExactSizeIterator<Item=LfrRow>,
        bits_per_value:u8,
        hash_key:&HasherKey,
        _max_threads: u8
    ) -> Option<MapCore<'a>> {
        if bits_per_value == 0 {
            /* force nblocks to 2 */
            return Some(MapCore{
                hash_key: *hash_key,
                bits_per_value: bits_per_value,
                nblocks: 2,
                blocks: Cow::Owned(Vec::new()),
            });
        }

        let nitems = iter.len();
        let mask = if bits_per_value == Response::BITS as u8 { !0 }
            else { (1<<bits_per_value)-1 };
        let nblocks = blocks_required(nitems);
        let nblocks_po2 = next_power_of_2_ge(nblocks);
        const BYTES_PER_VALUE : usize = (Response::BITS as usize) / 8;

        /* Create the blocks */
        let mut blocks : Vec<Matrix> = (0..nblocks).map(|_| {
            let mut ret = Matrix::new(0,BLOCKSIZE*8, bits_per_value as usize);
            ret.reserve_rows(BLOCKSIZE*8*2 * 5/4);
            ret
        }).collect();
        let mut row_ids : Vec<Vec<RowIdx>> = (0..nblocks).map(|_| {
            Vec::with_capacity(BLOCKSIZE*8*2 * 5/4)
        }).collect();
        
        /* Seed the items into the blocks.
         * PERF: not in parallel, because typically the cost of acquiring
         * the mutexes exceeds the cost of hashing.  But we could hash in
         * parallel and then bucket on one thread?  Or write parallel bucket
         * sort, heh.
         */
        let mut rowi=0;
        for row in iter {
            for i in [0,1] {
                let blki = row.block_positions[i] as usize;
                let aug_bytes = if i==0 { [0u8; BYTES_PER_VALUE] }
                    else { (row.augmented & mask).to_le_bytes() };
                blocks[blki].mut_add_row_as_bytes(&row.block_key[i].to_le_bytes(), &aug_bytes);
                row_ids[blki].push(rowi);
            }
            rowi += 1;
        }
        
        /* Create the block groups from the blocks */
        #[allow(unused_mut)] /* used in the threading case */
        let mut block_groups : Vec<BlockGroup> = (0..nblocks_po2*2).map(|_| BlockGroup::new()).collect();

        for (i,(blk,mut row_ids)) in blocks.into_iter().zip(row_ids.into_iter()).enumerate() {
            row_ids.push(RowIdx::MAX); /* Append guard rowidx */
            block_groups[2*i+1].fwd_solution.write((blk,row_ids,LfrBlock::BITS as usize));
        }
        for i in nblocks..nblocks_po2 {
            block_groups[2*i+1].fwd_solution.write((Matrix::new(0,0,0),Vec::new(),0));
        }

        /* Solve it */
        let naug = bits_per_value as usize;

        #[cfg(feature="threading")]
        {
            /* Threaded build */
            let parallelism = if _max_threads > 0 {
                _max_threads as usize
            } else {
                available_parallelism().map(|x| x.get()).unwrap_or(1)
            };
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
            block_groups = Arc::try_unwrap(arc_block_groups).ok().unwrap();
        }
        #[cfg(not(feature="threading"))]
        {
            Self::build(&block_groups, naug, nblocks, *hash_key)?;
        }

        /* Serialize to core */
        let mut core = vec![0u8; naug*nblocks*BLOCKSIZE];
        for blki in 0..nblocks {
            let bsol = block_groups[2*blki+1].bwd_solution.ctake().unwrap();
            debug_assert_eq!(bsol.rows, BLOCKSIZE*8);
            for aug in 0..naug {
                for byteoff in 0..BLOCKSIZE {
                    let mut byte = 0;
                    /* PERF: this could be faster, but do we care? */
                    for bit in 0..u8::BITS {
                        byte |= (bsol.get_aug_bit(byteoff*8 + bit as usize,aug) as u8) << bit;
                    }
                    core[(blki*naug+aug)*BLOCKSIZE+byteoff] = byte;
                }
            }
        }

        Some(MapCore{
            bits_per_value: bits_per_value,
            blocks: Cow::Owned(core),
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
        let naug = self.bits_per_value as usize;
        let p0 = BLOCKSIZE * naug * row.block_positions[0] as usize;
        let p1 = BLOCKSIZE * naug * row.block_positions[1] as usize;
        let [k0,k1] = row.block_key;
        let p0_bytes = &self.blocks[p0 .. p0+BLOCKSIZE*naug];
        let p1_bytes = &self.blocks[p1 .. p1+BLOCKSIZE*naug];
        for bit in 0..naug {
            /* Hopefully this all compiles to just a block access for p0 and for p1 */
            let get = (LfrBlock::from_le_bytes((&p0_bytes[bit*BLOCKSIZE..(bit+1)*BLOCKSIZE]).try_into().unwrap()) & k0)
                    ^ (LfrBlock::from_le_bytes((&p1_bytes[bit*BLOCKSIZE..(bit+1)*BLOCKSIZE]).try_into().unwrap()) & k1);
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
 * Compressed uniform static functions.
 * 
 * These objects work like [`CompressedMap`](crate::CompressedMap)s,
 * but they do not support arbitrary values as objects and do not
 * store their values.  Instead, they require the values to be integers
 * (or other objects implementing [`Into<u64>`](std::convert::Into) and
 * [`TryFrom<u64>`](std::convert::TryFrom)).
 * 
 * The design is efficient
 * if these integers are approximately uniformly random up to some bit size,
 * e.g. if they are random in `0..128`.  Unlike a [`CompressedMap`](crate::CompressedMap), a
 * [`CompressedRandomMap`] cannot take advantage of any bias in the distribution
 * of its values.
 *
 * [`CompressedRandomMap`] is building block for [`CompressedMap`](crate::CompressedMap).
 *
 * [`CompressedRandomMap`] doesn't implement [`Index`](core::ops::Index),
 * because it doesn't actually hold its values, so it can't return references
 * to them.
 */
#[derive(Eq,PartialEq,Ord,PartialOrd,Debug,Encode,Decode)]
pub struct CompressedRandomMap<'a,K,V> {
    /** Untyped map object, consulted after hashing. */
    pub(crate) core: MapCore<'a>,

    /** Phantom to hold the types of K,V */
    _phantom: PhantomData<(K,V)>
}

impl <'a,K,V> Clone for CompressedRandomMap<'a,K,V> {
    fn clone(&self) -> Self {
        CompressedRandomMap { core: self.core.clone(), _phantom:PhantomData::default() }
    }
}

/**
 * Approximate sets.
 *
 * These are like [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter),
 * except that are slower to construct, can't be modified once constructed,
 * and use about 30% less space.
 *
 * They store a compressed representation of a set `S` of objects.  From that
 * representation you can query whether an object is in the set.
 * 
 * There is a small, adjustable false positive probability, and no
 * false negatives.  That is, if you query an object that really was in the
 * set, you will always get `true`.  If you query an object not in the set,
 * you will usually get `false`, but not always: there is a small false
 * positive rate, according to the [`BuildOptions`].  There is a tradeoff:
 * the smaller the false positive rate, the more space is used by the
 * [`ApproxSet`].
 *
 * Internally, an [`ApproxSet`] is just a [`CompressedRandomMap`]
 * which maps all the elements of the set to zero.
 */
#[derive(Eq,PartialEq,Ord,PartialOrd,Debug,Encode,Decode)]
pub struct ApproxSet<'a,K> {
    /** Untyped map object, consulted after hashing. */
    core: MapCore<'a>,

    /** Phantom to hold the type of K */
    _phantom: PhantomData<K>,
}

impl <'a,K> Clone for ApproxSet<'a,K> {
    fn clone(&self) -> Self {
        ApproxSet { core: self.core.clone(), _phantom:PhantomData::default() }
    }
}

/**
 * Options to build a [`CompressedMap`](crate::CompressedMap),
 * [`CompressedRandomMap`] or [`ApproxSet`].
 * 
 * Implements `Default`, so you can get reasonable options
 * with `BuildOptions::default()`.
 */
#[derive(Copy,Clone,PartialEq,Eq,Debug,Ord,PartialOrd,Encode,Decode)]
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
     * If given for a [`CompressedRandomMap`], all values
     * will be truncated to that many least-significant bits.
     * If omitted, the bit length of the largest input value will be used.
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
     * Default: 0.
     */
    pub shift: u8,

    /**
     * Maximum threads to use for building.  If 0, use
     * available_parallelism().
     * 
     * Ignored if the threading feature isn't used.
     */
    pub max_threads: u8
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            key_gen : None,
            max_tries : 256,
            bits_per_value : None,
            try_num: 0,
            shift: 0,
            max_threads: 0
        }
    }
}

/** Next power of 2 >= x */
fn next_power_of_2_ge(x:usize) -> usize {
    if x==0 { return 1; }
    1 << (usize::BITS - (x-1).leading_zeros())
}

impl <'a,K:Hash,V:Copy> CompressedRandomMap<'a,K,V>
where Response:From<V> {
    /**
     * Build a uniform map.
     *
     * This function takes an iterable collection of items `(k,v)` and
     * constructs a compressed mapping.  If you query `k` on the compressed
     * mapping, `query` will return the corresponding `v`.  If you query any `k`
     * not included in the original list, the return value will be arbitrary.
     *
     * You can pass a [`HashMap<K,V>`](std::collections::HashMap),
     * [`BTreeMap<K,V>`](std::collections::BTreeMap) etc.  If you pass a
     * non-mapping type then be careful: any duplicate
     * `K` entries will cause the build to fail, possibly after a long time,
     * even if they have the same value associated.
     */
    pub fn build<'b, Collection>(map: &'b Collection, options: &mut BuildOptions) -> Option<Self>
    where &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          K:'b, V:'b,
          <&'b Collection as IntoIterator>::IntoIter : ExactSizeIterator,
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
            let iter1 = map.into_iter();
            let nblocks = blocks_required(iter1.len());
            let mut iter = iter1.map(|(k,v)| {
                let mut row = hash_object_to_row(&hkey,nblocks,k);
                row.augmented ^= Response::from(*v) >> options.shift;
                row
            });

            /* Solve it! (with type-independent code) */
            if let Some(solution) = MapCore::build_from_iter(
                &mut iter,
                bits_per_value,
                &hkey,
                options.max_threads
            ) {
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
     *
     * If (key,v) was included when building the map, then v will be returned.
     * Otherwise, an arbitrary value will be returned.
     */
    pub fn query(&self, key: &K) -> V 
    where V:From<Response> {
        self.core.query_hash(key).into()
    }

    /**
     * Query an item in the map.
     *
     * If (key,v) was included when building the map, then v will be returned.
     * Otherwise, an arbitrary value will be returned.
     */
    pub fn try_query(&self, key: &K) -> Option<V> 
    where V:TryFrom<Response> {
        self.core.query_hash(key).try_into().ok()
    }
}

impl <'a,K:Hash> ApproxSet<'a,K> {
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
    pub fn build<'b, Collection>(set: &'b Collection, options: &mut BuildOptions) -> Option<Self>
    where &'b Collection: IntoIterator<Item=&'b K>,
          K: 'b,
          <&'b Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Choose the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => Self::DEFAULT_BITS_PER_VALUE,
            Some(bpv) => bpv as u8
        };

        for try_num in options.try_num..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let iter1 = set.into_iter();
            let nblocks = blocks_required(iter1.len());
            let mut iter = iter1.map(|k| hash_object_to_row(&hkey,nblocks,k) );

            /* Solve it! (with type-independent code) */
            if let Some(solution) = MapCore::build_from_iter(
                &mut iter,
                bits_per_value,
                &hkey,
                options.max_threads
            ) {
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
     * There is a possibility of false positives, but not false negatives.
     */
    pub fn probably_contains(&self, key: &K) -> bool {
        self.core.query_hash(key) == 0
    }
}

use bincode::config::*;
/** Configuration to be used to encode / decode maps to binary file format */
pub const STD_BINCODE_CONFIG : Configuration<LittleEndian,Fixint,SkipFixedArrayLength,NoLimit>
    = bincode::config::standard()
    .with_little_endian()
    .with_fixed_int_encoding()
    .skip_fixed_array_length();

#[cfg(test)]
mod tests {
    use crate::uniform::{ApproxSet,CompressedRandomMap,BuildOptions,STD_BINCODE_CONFIG};
    use rand::{thread_rng,Rng,SeedableRng};
    use rand::rngs::StdRng;
    use std::collections::{HashMap,HashSet};
    use bincode::{encode_to_vec,decode_from_slice};

    /* Test map functionality */
    #[test]
    fn test_uniform_map() {
        let mut map = HashMap::new();
        for i in 0u32..10 {
            let mut seed = [0u8;32];
            seed[0..4].copy_from_slice(&i.to_le_bytes());
            let mut rng : StdRng = SeedableRng::from_seed(seed);
            for _j in 0..99*(i) {
                map.insert(rng.gen::<u64>(), rng.gen::<u8>());
            }
            let mut options = BuildOptions::default();
            options.key_gen = Some(seed[..16].try_into().unwrap());
            let crm = CompressedRandomMap::<u64,u8>::build(&map, &mut options).unwrap();

            for (k,v) in map.iter() {
                assert_eq!(crm.try_query(&k), Some(*v));
            }

            let ser = encode_to_vec(&crm, STD_BINCODE_CONFIG);
            assert!(ser.is_ok());
            let ser = ser.unwrap();
            let deser = decode_from_slice(&ser, STD_BINCODE_CONFIG);
            assert!(deser.is_ok());
            assert_eq!(crm, deser.unwrap().0);
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
            
            let approxset = ApproxSet::build(&set, &mut BuildOptions::default()).unwrap();
            for k in set.iter() {
                assert_eq!(approxset.probably_contains(&k), true);
            }

            let mut false_positives = 0;
            let ntries = 10000;
            let mu = ntries as f64 / 2f64.powf(approxset.core.bits_per_value as f64);
            for _ in 0..ntries {
                false_positives += approxset.probably_contains(&rng.gen::<u64>()) as usize;
            }
            // println!("{} false positives", false_positives);
            assert!((false_positives as f64) < mu + 4.*mu.sqrt());

            let ser = encode_to_vec(&approxset, STD_BINCODE_CONFIG);
            assert!(ser.is_ok());
            let ser = ser.unwrap();
            let deser = decode_from_slice(&ser, STD_BINCODE_CONFIG);
            assert!(deser.is_ok());
            assert_eq!(approxset, deser.unwrap().0);
        }
    }
}