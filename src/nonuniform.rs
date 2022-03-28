/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Nonuniform sparse linear map implementation.
 */

use crate::uniform::{MapCore,Map,Response,BuildOptions};
use std::collections::HashMap;
use core::marker::PhantomData;
use std::hash::Hash;
use std::cmp::{min,Ord,Ordering};

type Locator = u32;
type Plan = Locator;

/**
 * Sorted map: (lower bound, response).
 * All but at most one of widths
 * (i.e. lower bound #i+1 - lower bound i)
 * must be powers of two.
 */
type ResponseMap<V> = Vec<(Locator,V)>;

/** Next power of 2 that's less than x; minimum 1 */
fn floor_power_of_2(x:Locator) -> Locator {
    if x==0 { 1 } else {
        1<<(Locator::BITS - 1 - x.leading_zeros())
    }
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

    let mut total_width = 0;
    let mut items = Vec::new();
    for (k,v) in counts.iter() {
        let count = *v as u128;
        let ratio = ((count << Locator::BITS) as u128) / (total as u128);
        let width = floor_power_of_2(ratio as Locator);
        items.push((k,width,*v));
        total_width += width;
    }

    /* Sort them into priority order by "fit" */
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

    /* Sort by decreasing alignment, decreasing interval size, and then key */
    items.sort_by_key(|(k,w,_c)| (-(w.trailing_zeros() as i32),!w,*k));
    let mut total = 0;
    let mut plan = 0;
    let mut count = 0;
    for (k,w,c) in items {
        resp.push((total,(*k).clone()));
        value_map.insert(*k,count);
        interval_vec.push((c,total,total+(w-1)));
        count += 1;
        total += w;
        plan |= w;
        if (w & (w-1)) != 0 {
            plan |= (2 as Locator).wrapping_shl(Locator::BITS-1-w.leading_zeros());
        }
    }

    /* Done */
    (plan, value_map, interval_vec, resp)
}

pub struct NonUniformMap<K,V> {
    plan: Plan,
    response_map: ResponseMap<V>,
    salt: Vec<u8>,
    core: Vec<MapCore>,
    _phantom: PhantomData<K>
}

impl <K:Hash+Eq,V:Hash+Ord+Clone> NonUniformMap<K,V> {
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
     */
    pub fn build<'a, Collection>(map: &'a Collection, options: &BuildOptions) -> Option<NonUniformMap<K,V>>
    where for<'b> &'b Collection: IntoIterator<Item=(&'b K, &'b V)>,
          <&'a Collection as IntoIterator>::IntoIter : ExactSizeIterator
    {
        /* Count the items and formulate a plan */
        let mut counts = HashMap::new();
        for v in map.into_iter().map(|(_k,v)| v) {
            let counter = counts.entry(v).or_insert(0);
            *counter += 1;
        }
        let (plan, value_map, interval_vec, response_map) = formulate_plan(counts);
        let nphases = plan.count_ones() as usize;

        /* Record which bits are to be determined in each phase */
        let mut phase_bits = Vec::with_capacity(nphases);
        let mut plan_tmp = plan;
        while plan_tmp != 0 {
            let plan_tmp_2 = plan_tmp & !(plan_tmp-1);
            let before_plan  = (plan_tmp-1) & !plan_tmp;
            let before_plan2 = plan_tmp_2.wrapping_sub(1) & !plan_tmp_2;
            phase_bits.push(before_plan2 & !before_plan);
            plan_tmp = plan_tmp_2;
        }

        /* Which items must be put into which phases?
         * How many items are there in that phase?
         * Which is the index of non-power-of-2 item, if any?
         */
        let mut odd_man_out = usize::MAX;
        let mut n_omo = 0;
        let mut phase_to_resolve = Vec::with_capacity(interval_vec.len());
        let mut phase_item_counts = vec![0;nphases];
        for i in 0..interval_vec.len() {
            let (c,lo,hi) = interval_vec[i];
            let width = (hi-lo).wrapping_add(1);
            if width & width.wrapping_sub(1) != 0 {
                odd_man_out = i;
                n_omo = c;
                phase_to_resolve.push(u8::MAX);
            } else {
                for phase in 0..nphases {
                    if phase_bits[i] == lo^hi {
                        phase_to_resolve.push(phase as u8);
                        phase_item_counts[phase] += c;
                        break;
                    }
                }
            }
            assert!(phase_to_resolve.len() == i+1);
        }
    
        /* OK, for each phase, what's constrained that phase? */
        let mut constrained_in_phase : Vec<Vec<(&K, Response)>> = (0..nphases).map(
            |ph| Vec::with_capacity(phase_item_counts[ph])
        ).collect();
        let mut odd_men_out = Vec::with_capacity(n_omo);
        let mut current_values = vec![0 as Locator; n_omo];
        map.into_iter().for_each(|(k,v)| {
            let vi = value_map[v];
            if vi == odd_man_out {
                odd_men_out.push(k);
                let _ign = k;
            } else {
                let ph = phase_to_resolve[vi] as usize;
                let phlo = phase_bits[ph] & !(phase_bits[ph]-1);
                let (_c,lo,_hi) = interval_vec[vi];
                constrained_in_phase[vi as usize].push((k,(lo/phlo) as Response));
            }
        });

        /* Implement the plan! */
        let mut salt = vec![0u8];
        let mut core = Vec::new();
        let mut tries = options.max_tries;

        /* Phase by phase */
        while tries > 0 && salt.len() <= nphases {
            let phase = salt.len()-1;
            let phase_shift = phase_bits[phase].trailing_zeros();
            let mut phase_options = BuildOptions::default(); /* TODO */

            /* TODO: whoops, this is cumulative! 
             * Need to include items from past phases too!
             * Plan: sort items by phase, maybe, and then make
             * a FilteredVecIterator.
             */
            let phase_care = &constrained_in_phase[phase];
            unimplemented!("TODO: deal with OMO");

            if let Some(phase_map) = Map::build_from_vec(&phase_care, &mut phase_options) {
                salt.push(0);
                /* ... if we even care about OMO this phase ... */
                unimplemented!("TODO: update current_values");
            } else {
                salt[phase] = salt[phase].checked_add(1)?;
                tries -= 1;
            }
        }

        /* All done! */
        Some(NonUniformMap {
            plan: plan,
            response_map: response_map,
            salt: salt[1..nphases].to_vec(),
            core: core,
            _phantom: PhantomData::default()
        })
    }
}