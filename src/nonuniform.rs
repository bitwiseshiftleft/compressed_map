/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Nonuniform sparse linear map implementation.
 */

use crate::uniform::{MapCore,CompressedRandomMap,
    LfrBlock,BuildOptions,HasherKey,MapCoreSer,choose_key};
use crate::tilematrix::bitset::{BitSet,BitSetIterator};
use std::collections::HashMap;
use core::marker::PhantomData;
use std::hash::Hash;
use std::cmp::{min,Ord,Ordering};
use serde::{Serialize,Deserialize};

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
    -> (Plan, HashMap<&'a V, usize>, Vec<(usize,Locator,Locator)>, ResponseMap<V>)
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
        return (0,value_map,interval_vec,resp);
    }

    /* Count the weighted total number of items */
    let mut total = 0;
    for v in counts.values() { total += v; }
    debug_assert!(total > 0);

    let mut total_width = 0 as Locator;
    let mut items = Vec::new();
    for (k,v) in counts.iter() {
        let count = *v as u128;
        let ratio = ((count << Locator::BITS) as u128) / (total as u128);
        let width = floor_power_of_2(ratio as Locator);
        items.push((k,width,*v));
        total_width = total_width.wrapping_add(width);
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
    let mut remaining_width = total_width.wrapping_neg();
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
    (plan, value_map, interval_vec, resp)
}

/**
 * Compressed static functions.
 *
 * These functions are slightly slower than UniformMap, but they perform
 * efficiently even if not all values are equally likely.  They also accept
 * non-numeric types.
 */
#[derive(Eq,PartialEq,Ord,PartialOrd,Debug,Serialize,Deserialize)]
#[serde(bound(deserialize="V:Clone+Deserialize<'de>"))]
#[serde(bound(serialize="V:Clone+Serialize"))]
#[serde(try_from="CompressedMapSer<V>",into="CompressedMapSer<V>")]
pub struct CompressedMap<K,V> {
    plan: Plan,
    response_map: ResponseMap<V>,
    salt: Vec<u8>,
    core: Vec<MapCore>,
    _phantom: PhantomData<K>
}

impl <K,V> Clone for CompressedMap<K,V> where V:Clone {
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

impl <K:Hash+Eq+Sync,V:Hash+Ord+Clone> CompressedMap<K,V> {
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
    pub fn build<'a, Collection>(map: &'a Collection, options: &mut BuildOptions) -> Option<CompressedMap<K,V>>
    where for<'b> &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator
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

        let (plan, value_map, interval_vec, response_map) = formulate_plan(counts);
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
        let mut core : Vec<MapCore> = Vec::new();
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
            let phase_map = CompressedRandomMap::<K,Locator>::build::<FilteredVec<K>>(&phase_care, &mut phase_options)?;
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

    fn bsearch<'a>(&'a self, low: Locator, high: Locator) -> Option<&'a V> {
        let plow  = self.response_map.partition_point(|(begin,_v)| *begin <= low) - 1;
        if (plow == self.response_map.len() - 1)
            || (self.response_map[plow+1].0 > high) {
            Some(&self.response_map[plow].1)
        } else {
            None
        }
    }

    pub fn query<'a>(&'a self, key:&K) -> &'a V {
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

    /** Return the size of core storage, in bytes */
    pub fn core_size(&self) -> usize {
        let mut ret = 0;
        for ph in &self.core {
            ret += ph.blocks.len();
        }
        (ret * LfrBlock::BITS as usize) / 8
    }
}


/** Packed version for serialization */
#[derive(Eq,Ord,PartialEq,PartialOrd,Debug,Serialize,Deserialize)]
struct CompressedMapSer<V> {
    hash_key: HasherKey,
    plan: Plan,
    log_responses: Vec<u8>,
    salt: Vec<u8>,
    responses: Vec<V>,
    core: Vec<Vec<LfrBlock>>
}

impl <K,V> From<CompressedMap<K,V>> for CompressedMapSer<V> {
    fn from(map: CompressedMap<K,V>) -> Self {
        assert!(map.response_map.len() >= 1);
        let mut log_responses = Vec::with_capacity(map.response_map.len()-1);
        for i in 0..map.response_map.len()-1 {
            let delta = map.response_map[i+1].0 - map.response_map[i].0;
            log_responses.push(delta.leading_zeros() as u8+1);
        }
        let responses = map.response_map.into_iter().map(|(_loc,v)| v).collect();
        let hash_key = if map.core.len() == 0 {
            [0u8;16]
        } else {
            map.core[0].hash_key
        };
        CompressedMapSer {
            hash_key: hash_key,
            plan: map.plan,
            salt: map.salt,
            log_responses: log_responses,
            responses: responses,
            core: map.core.into_iter().map(|umap| umap.blocks).collect()
        }
    }
}

impl <K,V> TryFrom<CompressedMapSer<V>> for CompressedMap<K,V> {
    type Error = &'static str;
    fn try_from(ser: CompressedMapSer<V>) -> Result<CompressedMap<K,V>,&'static str> {
        /* Sanity check */
        let nphases = ser.plan.count_ones() as usize;
        if nphases > 0 && (nphases != ser.salt.len()+1) { return Err("ser.salt has the wrong length"); }
        if nphases != ser.core.len()   { return Err("ser.core has the wrong length"); }
        if ser.responses.len() != ser.log_responses.len() + 1 {
            return Err("ser responses len doesn't match log responses+1");
        }

        let mut response_map = Vec::with_capacity(ser.responses.len());
        let mut total : Locator = 0;
        for (i,response) in ser.responses.into_iter().enumerate() {
            if i < ser.log_responses.len() {
                let logr = ser.log_responses[i] as u32;
                if logr == 0 || logr > Locator::BITS {
                    return Err("invalid logr");
                }
                let r = 1 << (Locator::BITS - logr);
                response_map.push((total,response));
                total = total.checked_add(r).ok_or("responses must sum to < Locator::BITS")?;
            } else {
                response_map.push((total,response));
            }
        }


        let mut core = Vec::with_capacity(ser.core.len());
        let mut hashcur = ser.hash_key;
        let mut cur_plan = ser.plan;
        for (i,core_vec) in ser.core.into_iter().enumerate() {
            /* bits per value for the phase according to plan zeros */
            let next_plan = cur_plan & (cur_plan-1);
            let bpv = next_plan.trailing_zeros() - cur_plan.trailing_zeros();
            cur_plan = next_plan;
    
            /* reuse existing checker logic for MapCoreSer */
            let this_core = MapCore::try_from(MapCoreSer {
                hash_key: hashcur,
                bits_per_value: bpv as u8,
                blocks: core_vec,
            })?;

            /* update hash key according to salt */
            if i < ser.salt.len() {
                hashcur = choose_key(Some(hashcur), ser.salt[i] as usize);
            }
            core.push(this_core);
        }

        Ok(CompressedMap{
            plan: ser.plan,
            response_map: response_map,
            salt: ser.salt,
            core: core,
            _phantom: PhantomData::default()
        })
    }
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
    use rand::{Rng,SeedableRng};
    use rand::rngs::StdRng;
    use crate::nonuniform::{CompressedMap,BuildOptions};
    use std::collections::HashMap;
    use serde_json::{from_str, to_string};

    #[test]
    fn test_nonuniform_map() {
        assert!(CompressedMap::build(&HashMap::<u32,u32>::new(), &mut BuildOptions::default()).is_none());
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

            for (k,v) in map {
                assert_eq!(compressed_map.query(&k), &v);
            }

            /* test serialization */
            let ser = to_string(&compressed_map);
            assert!(ser.is_ok());
            let ser = ser.unwrap();
            let deser = from_str(&ser);
            // println!("{}", ser);
            assert!(deser.is_ok());
            assert_eq!(compressed_map, deser.unwrap()); 

        }
    }
}