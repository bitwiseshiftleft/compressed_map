/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Nonuniform sparse linear map implementation.
 */

use crate::uniform::{MapCore,CompressedRandomMap,BuildOptions,
    choose_key,BLOCKSIZE,STD_BINCODE_CONFIG,encode_u48,decode_u48,
    DefaultHasher,KeyedHasher128};
use crate::tilematrix::bitset::{BitSet,BitSetIterator};
use std::collections::HashMap;
use core::marker::PhantomData;
use core::hash::Hash;
use core::cmp::{min,max,Ord,Ordering};
use core::ops::Index;

use bincode::{Encode,Decode,BorrowDecode};
use bincode::enc::{Encoder};
use bincode::de::{BorrowDecoder};
use bincode::de::read::{Reader,BorrowReader};
use bincode::error::{EncodeError,DecodeError};
use bincode::enc::write::Writer;

use std::borrow::Cow;
use std::io::{Read,Error,ErrorKind,BufWriter,Write};
use std::fs::{File,OpenOptions};
use std::path::Path;

type Locator = u32;
type Plan = Locator;

/** Return the high bit of a locator.  Panics if 0. */
fn high_bit(x:Locator) -> u32 {
    Locator::BITS - 1 - x.leading_zeros()
}

/**
 * Sorted map: (lower bound, response).
 * All but at most one of widths
 * (i.e. lower bound #i+1 - lower bound i)
 * must be powers of two.
 */
type ResponseMap<V> = Vec<(Locator,V)>;

/** Next power of 2 that's less than x; minimum 1 */
fn floor_power_of_2(x:Locator) -> Locator {
    if x==0 { 1 } else { 1<<high_bit(x) }
}

/**
 * A "plan" is a bitfield describing which bits are determined
 * in which phases.  A 1 indicates a new phase, and a 0 indicates
 * that the bit is determined in the previous phase.
 *
 * Make a map of responses:
 *   HashMap: V -> value number
 *   Vec:     value number -> count and locator interval
 *   ResponseMap: locator interval -> V
 */
fn formulate_plan<'a, V:Ord+Clone+Hash>(counts: HashMap<&'a V,usize>)
    -> Option<
        (Plan, HashMap<&'a V, usize>, Vec<(usize,Locator,Locator)>, ResponseMap<V>)
    >
{
    /* Deal with special cases */
    let nitems = counts.len();
    let mut resp = Vec::new();
    let mut value_map = HashMap::new();
    let mut interval_vec = Vec::with_capacity(counts.len());
    if nitems <= 1 {
        for (x,c) in counts.iter() { // at most one item
            value_map.insert(*x,0);
            interval_vec.push((*c,0,Locator::MAX));
            resp.push((0 as Locator,(*x).clone()));
        }
        return Some((0,value_map,interval_vec,resp));
    }

    /* Count the weighted total number of items */
    let mut total = 0;
    for v in counts.values() { total += v; }
    debug_assert!(total > 0);

    /* Assign an initial interval size, which is its count/total, as a
     * 32-bit binary fraction, rounded down to the next power of 2.
     *
     * Because we always round down, the sum will always be in the interval
     * [2^31, 2^32] and the following code depends on this.
     *
     * Except: the fraction "0" (indicating < 1/2^32, since all counts are
     * actually nonzero) must round up to 1 instead.  In that case, the sum
     * could in fact overflow.  To fix this, we can "fudge" by halving all
     * nonzero interval sizes.
     *
     * TODO: test this fix.
     */
    let mut total_width;
    let mut items;
    let mut fudge = 0;
    loop {
        items = Vec::new();
        total_width = 0u64;
        for (k,v) in counts.iter() {
            let count = *v as u128;
            let ratio = (count << (Locator::BITS-fudge)) / (total as u128);
            let width = floor_power_of_2(ratio as Locator);
            items.push((k,width,*v));
            total_width += width as u64;
        }
        if Locator::BITS >= 64 || total_width <= (1u64 << Locator::BITS) {
            break;
        }
        fudge += 1;
        if fudge > 16 {
            return None;
        }
    }

    /* Sort them into priority order by "fit".  This is used to expand them
     * in case they sum to < the width.
     */
    fn compare_fit<T>(a1:&(&T,Locator,usize),a2:&(&T,Locator,usize)) -> Ordering {
        let (_k1,w1,c1) = *a1;
        let (_k2,w2,c2) = *a2;
        let score1 = (w1 as u128)*(c2 as u128);
        let score2 = (w2 as u128)*(c1 as u128);
        if score1 != score2 { return score1.cmp(&score2); }
        return (w1,c1).cmp(&(w2,c2)); // arbitrary
    }
    items.sort_by(compare_fit);

    /* Extend the intervals to the next power of 2 in priority order */
    /* 2u64 << bits-1 = 1u64<<bits, unless bits==64 in which case it's 0 (but without error) */
    let mut remaining_width = (2u64 << (Locator::BITS-1)).wrapping_sub(total_width) as Locator;
    for i in 0..items.len() {
        if remaining_width == 0 { break; }
        let expand = min(remaining_width, items[i].1);
        remaining_width -= expand;
        items[i].1 += expand;
    }

    /* Sort by hamming weight, decreasing alignment, and then key */
    items.sort_by_key(|(k,w,_c)| (w.count_ones(), -(w.trailing_zeros() as i32),*k));
    let mut total = 0;
    let mut plan = 0;
    let mut count = 0;
    for (k,w,c) in items {
        resp.push((total,(*k).clone()));
        value_map.insert(*k,count);
        interval_vec.push((c,total,total+(w-1)));
        count += 1;
        total = total.wrapping_add(w);
        if (w & (w-1)) == 0 {
            plan |= w;
        } else {
            /* Set the high bit */
            plan |= (1 as Locator).wrapping_shl(Locator::BITS-1-w.leading_zeros());
        }
    }

    /* Done */
    Some((plan, value_map, interval_vec, resp))
}

/// Compressed static functions.
/// 
/// These provide a compressed and efficiently queryable map from `K` to `V`,
/// somewhat like a [`HashMap<K,V>`](std::collections::hash_map).  However, unlike
/// a [`HashMap`](std::collections::hash_map), they do not store the keys.  So it
/// is not possible to list the keys of a [`CompressedMap`].  They also don't give
/// an error if when a key not in the map is queried: instead, a random value is
/// returned.  As a result, [`CompressedMap`]s can't be constructed from an empty
/// map.
/// 
/// [`CompressedMap`] implements [`Index`](core::ops::Index), so you can use
/// `map[k]` notation.
/// 
/// These maps are optimized for the case when there are only a few different
/// values, but millions of keys.
/// 
/// Asymptotically, for any distribution `D` of values, [`CompressedMap`]s use
/// at most 11% more space than the Shannon entropy of `D`, plus the size of
/// the values themselves.
#[derive(Debug)]
pub struct CompressedMap<'a,K,V,H=DefaultHasher> {
    plan: Plan,
    response_map: ResponseMap<V>,
    salt: Vec<u8>,
    core: Vec<MapCore<'a,H>>,
    _phantom: PhantomData<fn(K)>
}

impl <'a,K,V:PartialEq,H> PartialEq for CompressedMap<'a,K,V,H> {
    fn eq(&self,other:&Self) -> bool {
        self.plan == other.plan
        && self.response_map == other.response_map
        && self.salt == other.salt
        && self.core == other.core
    }
}
impl <'a,K,V:Eq,H> Eq for CompressedMap<'a,K,V,H> {}

impl <'a,K,V,H> Clone for CompressedMap<'a,K,V,H> where V:Clone {
    fn clone(&self) -> Self {
        CompressedMap{
            plan:self.plan,
            response_map: self.response_map.clone(),
            salt: self.salt.clone(),
            core: self.core.clone(),
            _phantom: PhantomData::default()
        }
    }
}

impl <'a,K:Hash,V,H:KeyedHasher128> CompressedMap<'a,K,V,H> {
    /**
     * Build a nonuniform map.
     *
     * The input must be non-empty.
     *
     * This function takes an iterable collection of items `(k,v)` and
     * constructs a compressed mapping.  If you query `k` on the compressed
     * mapping, `query` will return the corresponding `v`.  If you query any `k`
     * not included in the original list, the return value will be arbitrary.
     *
     * You can pass a `HashMap<T,u64>`, `BTreeMap<T,u64>` etc.  If you pass a
     * non-mapping type such as a `Vec` then be careful: any duplicate
     * `T` entries will cause the build to fail, possibly after a long time,
     * even if they have the same value associated.
     *
     * Ignores the BuildOptions' shift and bits_per_value.
     */
    pub fn build<'b, Collection>(map: &'b Collection, options: &mut BuildOptions) -> Option<Self>
    where &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          K: 'b, V: 'b+Hash+Ord+Clone,
          <&'b Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Count the items and formulate a plan */
        let mut counts = HashMap::new();
        map.into_iter().for_each(|(_k,v)| {
            let counter = counts.entry(v).or_insert(0);
            *counter += 1;
        });

        if counts.len() == 0 {
            return None;
        } else if counts.len() == 1 {
            let v = counts.keys().next().unwrap();
            return Some(CompressedMap {
                plan: 0,
                response_map: vec![(0,(*v).clone())],
                salt: vec![],
                core: vec![],
                _phantom: PhantomData::default()
            });
        }

        let (plan, value_map, interval_vec, response_map) = formulate_plan(counts)?;
        let nphases = plan.count_ones() as usize;

        /* Record which bits are to be determined in each phase */
        let mut phase_bits = Vec::with_capacity(nphases);
        let mut plan_tmp = plan;
        while plan_tmp != 0 {
            let plan_tmp_2 = plan_tmp & (plan_tmp-1);
            let before_plan  = (plan_tmp-1) & !plan_tmp;
            let before_plan2 = plan_tmp_2.wrapping_sub(1) & !plan_tmp_2;
            phase_bits.push(before_plan2 & !before_plan);
            plan_tmp = plan_tmp_2;
        }

        /* Which items must be put into which phases?
         * How many items are there in that phase?
         * Which is the index of non-power-of-2 item, if any?
         * (the "odd man out", or OMO)
         */
        let mut lo_omo = 0;
        let mut odd_man_out = usize::MAX;
        let mut phase_omo   = usize::MAX;
        let mut min_phase_affecting_omo = usize::MAX;
        let mut n_omo = 0;
        let mut phase_to_resolve = Vec::with_capacity(interval_vec.len());
        let mut phase_item_counts = vec![0;nphases];
        for i in 0..interval_vec.len() {
            let (c,lo,hi) = interval_vec[i];
            let width = (hi-lo).wrapping_add(1);
            if width & width.wrapping_sub(1) != 0 {
                odd_man_out = i;
                n_omo = c;
                lo_omo = lo;
                phase_to_resolve.push(u8::MAX);
                for phase in 0..nphases {
                    if phase_bits[phase] & width != 0 {
                        phase_omo = phase;
                        min_phase_affecting_omo = min(min_phase_affecting_omo,phase);
                    }
                }
            } else {
                for phase in 0..nphases {
                    if phase_bits[phase] & width != 0 {
                        phase_to_resolve.push(phase as u8);
                        phase_item_counts[phase] += c;
                        break;
                    }
                }
            }
            debug_assert!(phase_to_resolve.len() == i+1);
        }

        let mut phase_offsets = Vec::with_capacity(nphases);
        let mut total = n_omo;
        for phase in 0..nphases {
            phase_offsets.push(total);
            total += phase_item_counts[phase];
        }
    
        /* Sort by phase */
        let mut values_by_phase = Vec::new();
        let mut phase_offsets_cur = phase_offsets.clone();
        let mut current_values = vec![0 as Locator; n_omo];
        let mut omo_offset = 0;
        map.into_iter().for_each(|(k,v)| {
            if values_by_phase.len() == 0 {
                /* Need to initialize the vector with something;
                 * choose k arbitrarily.
                 */
                values_by_phase = vec![(k,0 as Locator); total];
            }
            let vi = value_map[v];
            let (_c,_lo,hi) = interval_vec[vi];
            if vi == odd_man_out {
                values_by_phase[omo_offset] = (k,hi);
                omo_offset += 1;
            } else {
                let ph = phase_to_resolve[vi] as usize;
                values_by_phase[phase_offsets_cur[ph]] = (k,hi);
                phase_offsets_cur[ph] += 1;
            }
        });

        /* Implement the plan! */
        let mut phase_care = FilteredVec {
            vec: values_by_phase,
            filter: BitSet::with_capacity(total)
        };
        let mut salt = Vec::new();
        let mut core : Vec<MapCore<H>> = Vec::new();
        let mut tries = options.try_num;

        /* Phase by phase */
        for phase in 0..nphases {
            let bits_this_phase = phase_bits[phase];
            let phase_shift = bits_this_phase.trailing_zeros();
            let phase_nbits = bits_this_phase.count_ones();
            let parent_key  = if phase == 0 { options.key_gen } else { Some(core[phase-1].hash_key) };
            let mut phase_options = BuildOptions {
                max_tries: min(options.max_tries - tries,255),
                try_num: 0,
                key_gen: parent_key,
                bits_per_value: Some(phase_nbits as u8),
                shift: phase_shift as u8,
                max_threads: options.max_threads
            };

            /* Set the values we care about */
            phase_care.filter.clear();
            phase_care.filter.union_with_range(omo_offset..phase_offsets_cur[phase]);
            if phase == phase_omo {
                /* Insert the ones that aren't above the beginning of the interval */
                for i in 0..omo_offset {
                    if current_values[i] < lo_omo {
                        phase_care.filter.insert(i);
                    }
                }
            } else if phase > phase_omo {
                phase_care.filter.union_with_range(0..omo_offset);
            }

            /* Solve the phase */
            let phase_map = CompressedRandomMap::<K,Locator,H>::build::<FilteredVec<K>>(&phase_care, &mut phase_options)?;
            tries += phase_options.try_num as usize;
            salt.push(phase_options.try_num as u8);
            if phase >= min_phase_affecting_omo && phase < phase_omo {
                for i in 0..omo_offset {
                    let (k,_) = phase_care.vec[i];
                    current_values[i] |= phase_map.try_query(k).unwrap() << phase_shift;
                }
            }
            core.push(phase_map.core);
        }

        options.try_num = tries;
        Some(CompressedMap {
            plan: plan,
            response_map: response_map,
            salt: salt[1..nphases].to_vec(),
            core: core,
            _phantom: PhantomData::default()
        })
    }

    fn bsearch<'b>(&'b self, low: Locator, high: Locator) -> Option<&'b V> {
        let plow  = self.response_map.partition_point(|(begin,_v)| *begin <= low) - 1;
        if (plow == self.response_map.len() - 1)
            || (self.response_map[plow+1].0 > high) {
            Some(&self.response_map[plow].1)
        } else {
            None
        }
    }

    #[inline]
    pub fn query<'b>(&'b self, key:&K) -> &'b V {
        let nphases = self.core.len();
        let mut locator = 0 as Locator;
        let mut plan = self.plan;
        if plan == 0 { return &self.response_map[0].1; }
        let mut known_mask = (plan-1) & !plan;

        /* The upper bits are the most informative.  However, in most cases the second-highest
        * map has more bits than the highest one, so it's actually fastest to start there.
        */
        if nphases >= 2 {
            let h1 = high_bit(plan);
            plan ^= 1<<h1;
            let h2 = high_bit(plan);
            let thisphase = self.core[nphases-2].query_hash(key) as Locator;
            known_mask |= (1<<h1) - (1<<h2);
            locator |= thisphase << h2;
            if let Some(result) = self.bsearch(locator, locator|!known_mask) {
                return result;
            }
        }

        plan = self.plan;
        for phase in (0..nphases).rev() {
            let h = high_bit(plan);
            plan ^= 1<<h;
            if phase+2 != nphases {
                let thisphase = self.core[phase].query_hash(key) as Locator;
                locator |= thisphase << h;
                known_mask |= ((1 as Locator)<<h).wrapping_neg();
                if let Some(result) = self.bsearch(locator, locator|!known_mask) {
                    return result;
                }
            }
        };
        
        unreachable!("CompressedRandomMap must have been constructed wrong; we should have a response by now")
    }

    /**
     * Take ownership, possibly copying the data
     *
     * This is useful if you created the object using
     * [`borrow_decode`](bincode::BorrowDecode::borrow_decode), but want to
     * own the data independently.
     */
    pub fn take_ownership<'b>(self) -> CompressedMap<'b,K,V,H> {
        CompressedMap { 
            plan: self.plan,
            response_map: self.response_map,
            salt: self.salt,
            core: self.core.into_iter().map(|c| c.take_ownership()).collect(),
            _phantom: PhantomData::default()
        }
    }

    /**
     * Write the map to a new file.
     *
     * Raise an error if the file exists, or cannot be created, or if an I/O
     * error occurs.
     */
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Error>
    where V: Encode {
        let file = OpenOptions::new().create_new(true).write(true).open(path)?;
        let mut writer = BufWriter::new(file);
        bincode::encode_into_std_write(self, &mut writer, STD_BINCODE_CONFIG).map_err(
            |e| match e {
                EncodeError::Io{ error:e, index:_s } => e,
                EncodeError::Other(s) => Error::new(ErrorKind::Other, s),
                _ => Error::new(ErrorKind::Other, e.to_string()),
        })?;
        writer.flush()
    }

    /**
     * Read a map from a file.
     *
     * Return an error if the file doesn't exist or is not readable, if an I/O error
     * occurs, if the map is corrupt, or if there are bytes left at the end of the
     * file after decoding.
     */
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error>
    where V: Decode {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let (unowned,sz) : (CompressedMap<K,V,H>,usize)
            = bincode::decode_from_slice(&buf, STD_BINCODE_CONFIG)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        if sz < buf.len() {
            Err(Error::new(ErrorKind::Other, "bytes left over on read_from_file".to_string()))
        } else {
            Ok(unowned.take_ownership())
        }
    }
}

const MAGIC: &[u8;4] = b"cnm1";

impl <'a,K,V:Encode,H> Encode for CompressedMap<'a,K,V,H> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(MAGIC, encoder)?;

        assert!(self.response_map.len() >= 1);
        let mut log_responses = Vec::with_capacity(self.response_map.len()-1);
        for i in 0..self.response_map.len()-1 {
            let delta = self.response_map[i+1].0 - self.response_map[i].0;
            log_responses.push(delta.leading_zeros() as u8+1);
        }
        let hash_key = if self.core.len() == 0 {
            [0u8;16]
        } else {
            self.core[0].hash_key
        };

        Encode::encode(&log_responses, encoder)?;
        for (_l,v) in &self.response_map {
            Encode::encode(v, encoder)?;
        }

        Encode::encode(&hash_key, encoder)?;
        Encode::encode(&self.plan, encoder)?;
        encoder.writer().write(&self.salt)?;
        for core in &self.core {
            encode_u48(core.nblocks,encoder)?;
        }
        for core in &self.core {
            encoder.writer().write(&core.blocks.as_ref())?;
        }
        Ok(())
    }
}

impl <'a,'de:'a,K,V,H:KeyedHasher128> BorrowDecode<'de> for CompressedMap<'a,K,V,H> where V: BorrowDecode<'de> {
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        /* Decode the response map */
        fn err<Nope>(descr: &'static str) -> Result<Nope, DecodeError> {
            Err(DecodeError::OtherString(descr.to_string()))
        }
        let magic : [u8;4] = Decode::decode(decoder)?;
        if &magic != MAGIC {
            return err("magic value mismatch");
        }

        /* First: log_responses and responses */
        let log_responses : Vec<u8> = Decode::decode(decoder)?;
        let mut responses : Vec<V> = Vec::with_capacity(log_responses.len()+1);
        for _ in 0..log_responses.len()+1 {
            responses.push(BorrowDecode::borrow_decode(decoder)?);
        }
        let mut response_map = Vec::with_capacity(responses.len());
        let mut total : Locator = 0;
        for (i,response) in responses.into_iter().enumerate() {
            if i < log_responses.len() {
                let logr = log_responses[i] as u32;
                if logr == 0 || logr > Locator::BITS {
                    return err("invalid logr");
                }
                let r = 1 << (Locator::BITS - logr);
                response_map.push((total,response));
                total = total.checked_add(r)
                    .ok_or(DecodeError::OtherString("responses must sum to < Locator::BITS".to_string()))?;
            } else {
                response_map.push((total,response));
            }
        }


        /* Hash_key and plan */
        let hash_key = Decode::decode(decoder)?;
        let plan : Locator = Decode::decode(decoder)?;
        let nphases = plan.count_ones() as usize;
        
        /* Salt and nblocks per phase */
        let len_salt = max(1,nphases)-1;
        let mut salt = vec![0u8;len_salt];
        decoder.reader().read(&mut salt)?;
        let mut nblocks_per_phase : Vec<usize> = Vec::with_capacity(nphases);
        for _phase in 0..nphases {
            let nblocks = decode_u48(decoder)?;
            if nblocks < 2 { return err("must have at least 2 nblocks"); }
            nblocks_per_phase.push(nblocks);
        }


        /* The vectors for the cores */
        if nphases > 0 && (nphases != salt.len()+1) { return err("salt has the wrong length"); } /* TODO: unredund this */
        let mut core : Vec<MapCore<H>> = Vec::with_capacity(nphases);
        let mut hashcur = hash_key;
        let mut cur_plan = plan;
        for phase in 0..nphases {
            let nblocks = nblocks_per_phase[phase];
            let next_plan = cur_plan & (cur_plan-1);
            let bpv = next_plan.trailing_zeros() - cur_plan.trailing_zeros();
            cur_plan = next_plan;

            /* multiply and check for overflow */
            let mul1 : usize = nblocks.checked_mul(BLOCKSIZE)
                .ok_or(DecodeError::OtherString("overflow on multiply".to_string()))?;
            let mul2 : usize = mul1.checked_mul(bpv as usize)
                .ok_or(DecodeError::OtherString("overflow on multiply".to_string()))?;

            /* Read the blocks in straight */
            let borrowed = decoder.borrow_reader().take_bytes(mul2)?;
            core.push( MapCore {
                hash_key: hashcur,
                bits_per_value: bpv as u8,
                nblocks: nblocks,
                blocks: Cow::Borrowed(borrowed),
                _phantom: PhantomData::default()
            });

            /* update hash key according to salt */
            if phase < salt.len() {
                hashcur = choose_key::<H>(Some(hashcur), salt[phase] as usize);
            }
        }

        Ok(CompressedMap{
            plan: plan,
            response_map: response_map,
            salt: salt,
            core: core,
            _phantom: PhantomData::default()
        })
    }
}

impl <'a,K:Hash,V> Index<&K> for CompressedMap<'a,K,V> where {
    type Output = V;
    fn index(&self, index: &K) -> &V { self.query(index) }
}

impl <'a,K:Hash,V> Index<K> for CompressedMap<'a,K,V> where {
    type Output = V;
    fn index(&self, index: K) -> &V { self.query(&index) }
}

/** Utility: vector with bitset selecting which of its elements are iterated over. */
struct FilteredVec<'a,T> {
    vec: Vec<(&'a T,Locator)>,
    filter: BitSet
}

struct FilteredVecIterator<'a,T> {
    vec: &'a Vec<(&'a T,Locator)>,
    bsi: BitSetIterator<'a>
}

impl <'a,T> Iterator for FilteredVecIterator<'a,T> {
    type Item = (&'a T, &'a Locator);
    fn size_hint(&self) -> (usize,Option<usize>) { self.bsi.size_hint() }
    fn next(&mut self) -> Option<(&'a T, &'a Locator)> {
        let i = self.bsi.next()?;
        let (k,v) = &self.vec[i];
        Some((k,&v))
    }
}

impl <'a,T> ExactSizeIterator for FilteredVecIterator<'a,T> {}

impl <'a,'b,T> IntoIterator for &'a FilteredVec<'b,T> {
    type Item = (&'a T, &'a Locator);
    type IntoIter = FilteredVecIterator<'a,T>;
    fn into_iter(self) -> FilteredVecIterator<'a,T> {
        FilteredVecIterator { vec: &self.vec, bsi: self.filter.into_iter() }
    }
}


#[cfg(test)]
mod tests {
    use crate as compressed_map;

    #[test]
    fn test_nonuniform_map() {
        use rand::{Rng,SeedableRng};
        use rand::rngs::StdRng;
        use crate::nonuniform::{CompressedMap,BuildOptions};
        use crate::STD_BINCODE_CONFIG;
        use std::collections::HashMap;
        use bincode::{encode_to_vec,decode_from_slice};

        assert!(CompressedMap::<_,_>::build(&HashMap::<u32,u32>::new(), &mut BuildOptions::default()).is_none());
        for i in 0u32..100 {
            let mut seed = [0u8;32];
            seed[0..4].copy_from_slice(&i.to_le_bytes());
            let mut rng : StdRng = SeedableRng::from_seed(seed);
            let mut map = HashMap::new();
            let n_items = i/10+1;
            let pr_splits : Vec<f64> = (0..n_items).map(|_| rng.gen_range(0.0..1.0)).collect();

            let nitems = 1000;
            for _ in 0..nitems {
                let mut v = n_items-1;
                for (i,p) in (&pr_splits).into_iter().enumerate() {
                    if rng.gen_range(0.0..1.0) < *p {
                        v = i as u32;
                        break;
                    }
                }
                map.insert(rng.gen::<u32>(), v);
            }

            let mut options = BuildOptions::default();
            options.key_gen = Some(seed[..16].try_into().unwrap());

            let compressed_map = CompressedMap::build(&map, &mut options).unwrap();
            for (k,v) in map { assert_eq!(compressed_map[k], v); }

            /* test serialization */
            let ser = encode_to_vec(&compressed_map, STD_BINCODE_CONFIG);
            assert!(ser.is_ok());
            let ser = ser.unwrap();
            let deser = decode_from_slice(&ser, STD_BINCODE_CONFIG);
            assert!(deser.is_ok());
            assert_eq!(compressed_map, deser.unwrap().0);
        }
    }

    #[test]
    fn simple_test_nonuniform_map() {
        // Import relevant libraries
        use rand::{Rng,thread_rng,distributions::{Bernoulli, Distribution}};
        use compressed_map::{
            CompressedMap,BuildOptions,
            serialized_size,STD_BINCODE_CONFIG
        };
        use std::collections::HashMap;

        // Set up the RNG
        let distribution = Bernoulli::new(0.05).unwrap();
        let mut rng = thread_rng();

        // Create a map with 100k items, about 95% "no" : 5% "yes"
        let nitems = 100000;
        let mut map = HashMap::new();
        for _ in 0..nitems {
            map.insert(rng.gen::<u64>(),distribution.sample(&mut rng));
        }

        // Compress the map
        let cmap = CompressedMap::<'_,u64,bool>::build(&map,
            &mut BuildOptions::default()).unwrap();

        // Query the compressed map: the answer is the same as for the hashmap
        // Also count the true proportion so we can compute the Shannon limit
        let mut nyes = 0;
        for (key,value) in &map {
            assert_eq!(value,cmap.query(&key));
            nyes += *value as u64;
        }
        let p = nyes as f64 / nitems as f64; 

        // How big is the map?
        let hash_sersize = serialized_size(&map,STD_BINCODE_CONFIG).unwrap();
        let sersize = serialized_size(&cmap,STD_BINCODE_CONFIG).unwrap();
        let shannon = nitems as f64 * -(p*p.log2() + (1.-p)*(1.-p).log2());
        println!("hashmap={} bytes, cmap={} bytes, ratio={:0.1}, {:0.2} bits/key",
            hash_sersize, sersize, hash_sersize as f64/sersize as f64,
            sersize as f64 * 8. / nitems as f64);
        println!("Shannon limit for {:0.2}%={} bytes, overhead={:0.2}%",
            p*100., (shannon/8.) as u64, (sersize as f64*8. / shannon - 1.) * 100.);
            
        // Example output:
        // hashmap=900008 bytes, cmap=3952 bytes, ratio=227.7, 0.32 bits/key
        // Shannon limit for 5.03%=3596 bytes, overhead=9.87%
    }
}