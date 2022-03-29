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
pub(crate) type LfrBlock = u32;

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

/** A block or group of blocks for the hierarchical solver */
struct BlockGroup {
    contents:   Matrix,
    row_ids:    Vec<RowIdx>, // with RowIdx::MAX appended for convenience
    solution:   Systematic,
    empty:      bool
}

impl BlockGroup {
    fn new(cols: usize, aug: usize, empty:bool) -> BlockGroup {
        BlockGroup {
            contents: Matrix::new(0, cols, aug),
            row_ids: Vec::new(),
            solution: Systematic::identity(0),
            empty: empty
        }
    }
}

/**
 * Forward solve step.
 * Merge the resolvable rows of the left and right BlockGroups, then
 * project them out of the remaining ones.  Clears the contents of left and right,
 * but not the solution.
 *
 * Fails if the block group doesn'K have full row-rank.
 */
fn forward_solve(left:&mut BlockGroup, right:&mut BlockGroup) -> Option<BlockGroup> {
    if right.empty {
        return Some(replace(left, BlockGroup::new(0,0,true)));
    } else if left.empty {
        return Some(replace(right, BlockGroup::new(0,0,true)));
    }

    /* Mergesort the row IDs */
    let mut left_to_merge = BitSet::with_capacity(left.contents.rows);
    let mut right_to_merge = BitSet::with_capacity(right.contents.rows);
    let mut interleave_left = BitSet::with_capacity(left.contents.rows+right.contents.rows);
    let mut row_ids = Vec::new();
    row_ids.reserve(left.contents.rows + right.contents.rows); // over-reserve but this isn'K the long pole
    let (mut l, mut r, mut i) = (0,0,0);

    /* There should always be something in the left and right group,
        * because we appended RowIdx::MAX
        */
    while (l < left.contents.rows) || (r < right.contents.rows) {
        if left.row_ids[l] == right.row_ids[r] {
            left_to_merge.insert(l);
            right_to_merge.insert(r);
            l += 1;
            r += 1;
        } else if left.row_ids[l] < right.row_ids[r] {
            row_ids.push(left.row_ids[l]);
            interleave_left.insert(i);
            l += 1;
            i += 1;
        } else {
            row_ids.push(right.row_ids[r]);
            r += 1;
            i += 1;
        }
    }
    row_ids.push(RowIdx::MAX);

    /* Memory efficiency: drop the row IDs */
    left.row_ids.clear();
    left.row_ids.shrink_to_fit();
    right.row_ids.clear();
    right.row_ids.shrink_to_fit();

    /* Sort out the rows */
    let (lmerge,lkeep) = left.contents.partition_rows(&left_to_merge);
    left.contents.clear();
    let (rmerge,rkeep) = right.contents.partition_rows(&right_to_merge);
    right.contents.clear();

    /* Merge and solve */
    let mut merged = lmerge.append_columns(&rmerge);
    let solution = merged.systematic_form()?;
    let lproj = solution.project_out(&lkeep, true);
    let rproj = solution.project_out(&rkeep, false);
    let interleaved = lproj.interleave_rows(&rproj, &interleave_left);

    Some(BlockGroup { contents: interleaved, row_ids: row_ids, solution: solution, empty:false })
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
        *center = BlockGroup::new(0,0,true);
    } else {
        (left.contents, right.contents) = contents.split_at_row(left.solution.rhs.cols_main);
        *center = BlockGroup::new(0,0,true);
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
pub(crate) struct MapCore {
    /** The SipHash key used to hash inputs. */
    pub(crate) hash_key: HasherKey,

    /** The number of bits stored per (key,value) pair. */
    pub bits_per_value: usize,

    /** The number of blocks per bit */
    pub nblocks: usize,

    pub(crate) blocks: Vec<LfrBlock>,
}

/** Simple serializer for bytes.
 * 
 * On serialize, first get a length, then serialize into a byte
 * array of that length.
 * 
 * Deserialize from byte arrays that are exactly the right size.
 * Return None for any failure.
 */
pub trait SimpleSerialize where Self:Sized {
    /** Return the serialized size of self, in bytes */
    fn ser_size(&self) -> usize;

    /** Serialize self into an already-allocated array of bytes */
    fn serialize_into(&self, bytes: &mut[u8]);

    /** Deserialize self from a string of bytes, or fail */
    fn deserialize(bytes: &[u8]) -> Option<Self>;

    /** Default: serialize self into a Vec and return it */
    fn serialize(&self) -> Vec<u8> {
        let mut ret = vec![0u8;self.ser_size()];
        self.serialize_into(&mut ret);
        ret
    }
}

impl SimpleSerialize for Vec<LfrBlock> {
    fn ser_size(&self) -> usize { self.len() * BLOCKSIZE }
    fn serialize_into(&self, bytes: &mut[u8]) {
        for (i,b) in self.into_iter().enumerate() {
            bytes[i*BLOCKSIZE..(i+1)*BLOCKSIZE].copy_from_slice(&b.to_le_bytes());
        }
    }
    fn deserialize(bytes:&[u8]) -> Option<Self> {
        if (bytes.len() % BLOCKSIZE) != 0 { return None; }
        let blocks = bytes.len()/BLOCKSIZE;
        let mut result = vec![0;blocks];
        for i in 0..blocks {
            result[i] = LfrBlock::from_le_bytes(bytes[i*BLOCKSIZE..(i+1)*BLOCKSIZE].try_into().unwrap());
        }
        Some(result)
    }
}

impl SimpleSerialize for MapCore {
    fn ser_size(&self) -> usize { 20 + self.blocks.len() * BLOCKSIZE }
    fn serialize_into(&self, bytes: &mut[u8]) {
        bytes[0..3].copy_from_slice("MC1".as_bytes());
        bytes[3] = self.bits_per_value as u8;
        bytes[4..20].copy_from_slice(&self.hash_key);
        self.blocks.serialize_into(&mut bytes[20..]);
    }
    fn deserialize(bytes:&[u8]) -> Option<Self> {
        if bytes.len() < 20 { return None; }
        if bytes[0..3] != *"MC1".as_bytes() { return None; }
        let bits_per_value = bytes[3] as usize;
        if bits_per_value > Response::BITS as usize { return None; }

       let nblocks = if bits_per_value == 0 {
            if bytes.len() != 20 { return None; }
            2
        } else {
            if (bytes.len()-20) % (bits_per_value * BLOCKSIZE) != 0 { return None; }
            (bytes.len()-20) / (bits_per_value * BLOCKSIZE)
        };
        if nblocks == 1 { return None; }

        let blocks = Vec::<LfrBlock>::deserialize(&bytes[20..])?;
        Some(MapCore {
            hash_key: bytes[4..20].try_into().unwrap(),
            bits_per_value: bits_per_value,
            nblocks: nblocks,
            blocks: blocks
        })
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
    fn build(mut blocks: Vec<BlockGroup>, nblocks:usize, hash_key: &HasherKey) -> Option<MapCore> {
        let mut halfstride = 1;
        while halfstride < nblocks {
            let mut pos = 2*halfstride;
            while pos < blocks.len() {
                let (l,r) = blocks.split_at_mut(pos);
                blocks[pos] = forward_solve(&mut l[pos-halfstride], &mut r[halfstride])?;
                pos += 4*halfstride;
            }
            halfstride *= 2;
        }

        /* Initialize solution at random */
        let naug = blocks[halfstride].solution.rhs.cols_aug;
        let mut seed = Matrix::new(0,0,naug);
        let empty = [];
        for row in 0..blocks[halfstride].solution.rhs.cols_main {
            seed.mut_add_row_as_bytes(&empty, &MapCore::seed_row(hash_key, naug, row));
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
            blocks: core,
            nblocks: nblocks,
            hash_key: *hash_key
        })
    }

    /** Try once to build from an iterator. */
    fn build_from_iter(
        iter: &mut dyn ExactSizeIterator<Item=LfrRow>,
        bits_per_value:usize,
        hash_key:&HasherKey
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
        let mask = if bits_per_value == Response::BITS as usize { !0 }
            else { (1<<bits_per_value)-1 };
        let nblocks = blocks_required(nitems);
        let nblocks_po2 = next_power_of_2_ge(nblocks);
        const BYTES_PER_VALUE : usize = (Response::BITS as usize) / 8;

        /* Create the blocks */
        let mut blocks : Vec<BlockGroup> = (0..nblocks_po2*2).map(|i|
            if ((i&1) != 0) && i < 2*nblocks {
                let mut ret = BlockGroup::new(BLOCKSIZE*8, bits_per_value, false);
                ret.row_ids.reserve(BLOCKSIZE*8*2 * 5/4);
                ret.contents.reserve_rows(BLOCKSIZE*8*2 * 5/4);
                ret
            } else {
                BlockGroup::new(0,0,true)
            }
        ).collect();

        let mut rowi=0;
        for row in iter {
            for i in [0,1] {
                let blki = row.block_positions[i] as usize;
                let blk = &mut blocks[2*blki+1];
                let aug_bytes = if i==0 { [0u8; BYTES_PER_VALUE] }
                    else { (row.augmented & mask).to_le_bytes() };
                blk.contents.mut_add_row_as_bytes(&row.block_key[i].to_le_bytes(), &aug_bytes);
                blk.row_ids.push(rowi);
            }
            rowi += 1;
        }

        /* Append guard rowidx */
        for blk in &mut blocks { blk.row_ids.push(RowIdx::MAX); }

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
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Debug)]
pub struct CompressedRandomMap<K,V> {
    /** Untyped map object, consulted after hashing. */
    pub(crate) core: MapCore,

    /** Phantom to hold the types of K,V */
    _phantom: PhantomData<(K,V)>
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
#[derive(Eq,PartialEq,Ord,PartialOrd,Clone,Debug)]
pub struct ApproxSet<K> {
    /** Untyped map object, consulted after hashing. */
    core: MapCore,

    /** Phantom to hold the type of K */
    _phantom: PhantomData<K>,
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

impl <K,V> SimpleSerialize for CompressedRandomMap<K,V> {
    fn ser_size(&self) -> usize { self.core.ser_size() }
    fn serialize_into(&self, bytes: &mut[u8]) { self.core.serialize_into(bytes); }
    fn deserialize(bytes: &[u8]) -> Option<Self> {
        let core = MapCore::deserialize(bytes)?;
        Some(CompressedRandomMap { core: core, _phantom:PhantomData::default() })
    }
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

impl <K:Hash,V:Copy> CompressedRandomMap<K,V>
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
    where for<'b> &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator,
    {
        /* Get the number of bits required */
        let bits_per_value = match options.bits_per_value {
            None => {
                let mut range : Response = 0;
                for v in map.into_iter().map(|(_k,v)| v) { range |= Response::from(*v); }
                (Response::BITS - range.leading_zeros()) as usize
            },
            Some(bpv) => bpv as usize
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
            if let Some(solution) = MapCore::build_from_iter(&mut iter, bits_per_value, &hkey) {
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

impl <K> SimpleSerialize for ApproxSet<K> {
    fn ser_size(&self) -> usize { self.core.ser_size() }
    fn serialize_into(&self, bytes: &mut[u8]) { self.core.serialize_into(bytes); }
    fn deserialize(bytes: &[u8]) -> Option<Self> {
        let core = MapCore::deserialize(bytes)?;
        Some(ApproxSet { core: core, _phantom:PhantomData::default() })
    }
}

impl <K:Hash> ApproxSet<K> {
    /** Default bits per value if none is specified. */
    const DEFAULT_BITS_PER_VALUE : usize = 8;

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
            Some(bpv) => bpv as usize
        };

        for try_num in options.try_num..options.max_tries {
            let hkey = choose_key(options.key_gen, try_num);
            let iter1 = set.into_iter();
            let nblocks = blocks_required(iter1.len());
            let mut iter = iter1.map(|k| hash_object_to_row(&hkey,nblocks,k) );

            /* Solve it! (with type-independent code) */
            if let Some(solution) = MapCore::build_from_iter(&mut iter, bits_per_value, &hkey) {
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
    use crate::uniform::{ApproxSet,CompressedRandomMap,BuildOptions,SimpleSerialize};
    use rand::{thread_rng,Rng};
    use std::collections::{HashMap,HashSet};

    /* Test map functionality */
    #[test]
    fn test_uniform_map() {
        let mut rng = thread_rng();
        let mut map = HashMap::new();
        for i in 0..10 {
            for _j in 0..99*i {
                map.insert(rng.gen::<u64>(), rng.gen::<u8>());
            }
            let hiermap = CompressedRandomMap::<u64,u8>::build(&map, &mut BuildOptions::default()).unwrap();

            for (k,v) in map.iter() {
                assert_eq!(hiermap.try_query(&k), Some(*v));
            }

            let ser = hiermap.serialize();
            let deser = CompressedRandomMap::<u64,u8>::deserialize(&ser);
            assert!(deser.is_some());
            assert_eq!(hiermap, deser.unwrap()); 
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
            
            let ser = hiermap.serialize();
            let deser = ApproxSet::<u64>::deserialize(&ser);
            assert!(deser.is_some());
            assert_eq!(hiermap, deser.unwrap()); 
        }
    }
}