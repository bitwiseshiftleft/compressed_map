/*
 * @file uniform.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Nonuniform sparse linear map implementation.
 */

use crate::uniform;
use std::collections::HashMap;
use std::cmp::{min,Ord,Ordering};

type Locator = u32;
type Plan = Locator;

struct ResponseMap<T> {
    responses : Vec<(Locator,T)>
}

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
 * Formulate a response map and a 
 */
fn formulate_plan<T:Clone>(counts: &HashMap<T,usize>) -> (ResponseMap<T>, Plan) {
    /* Deal with special cases */
    let nitems = counts.len();
    if nitems == 0 {
        return (ResponseMap{responses:vec![]},0);
    } else if nitems == 1 {
        let mut resp = Vec::new();
        for x in counts.keys() { resp.push((0,x.clone())); }
        return (ResponseMap{responses:resp},0);
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

    /* Extend the intervals in priority order */
    let mut remaining_width = total_width.wrapping_neg();
    for i in 0..items.len() {
        if remaining_width == 0 { break; }
        let expand = min(remaining_width, items[i].1);
        remaining_width -= expand;
        items[i].1 += expand;
    }

    /* Sort by alignment */
    items.sort_by_key(|(_k,w,c)| (-(w.trailing_zeros() as i32),*c));
    let mut total = 0;
    let mut plan = 0;
    let mut resp = Vec::new();
    for (k,w,_c) in items {
        resp.push((total,k.clone()));
        total += w;
        plan |= w;
        if (w & (w-1)) != 0 {
            plan |= (2 as Locator).wrapping_shl(Locator::BITS-1-w.leading_zeros());
        }
    }

    /* Done */
    (ResponseMap{responses:resp}, plan)
}
