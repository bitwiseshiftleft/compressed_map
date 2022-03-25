/*
 * @file bitset.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * A few operations on bit sets, because for some reason the
 * bit-set crate is bottlenecking?
 */

use core::ops::Range;

pub struct BitSet {
    set: Vec<u64>
}

impl BitSet {
    /** Initialize an empty set */
    pub fn with_capacity(capacity:usize) -> Self {
        BitSet { set : vec![0; (capacity+63)/64] }
    }

    /** Clear and free self */
    #[allow(dead_code)]
    pub fn free(&mut self) {
        self.set.resize(0,0);
        self.set.shrink_to_fit();
    }

    /** Set a bit in the set */
    #[inline(always)]
    pub fn insert(&mut self, x:usize) {
        self.set[x/64] |= 1<<(x%64);
    }

    /** Remove a bit in the set */
    #[inline(always)]
    #[allow(dead_code)]
    pub fn remove(&mut self, x:usize) {
        self.set[x/64] &= !(1<<(x%64));
    }

    /** Check a bit in the set */
    #[inline(always)]
    pub fn contains(&self, x:usize) -> bool {
        x/64 < self.set.len() && (self.set[x/64] & 1<<(x%64)) != 0
    }

    /** Check a bit in the set */
    pub fn len(&self) -> usize {
        let mut ret = 0;
        for x in &self.set { ret += x.count_ones() as usize };
        ret
    }

    /** Create a set from a given range */
    pub fn from_range(capacity:usize, range:Range<usize>) -> Self {
        let mut ret = BitSet { set : vec![0; (capacity+63)/64] };
        for i in range.start/64 .. (range.end + 63) / 64 {
            let mut mask = !0;
            if i*64 < range.start {
                mask &= (1u64<<(range.start-i*64)).wrapping_neg();
            }
            if (i+1)*64 > range.end {
                mask &= (1u64 << (range.end-i*64)) - 1;
            }
            ret.set[i] = mask;
        }
        ret
    }

    /** Create a set from a given range */
    pub fn count_within(&self, range:Range<usize>) -> usize {
        let mut ret = 0;
        for i in range.start / 64 .. (range.end + 63) / 64 {
            let mut mask = !0;
            if i*64 < range.start {
                mask &= (1u64<<(range.start%64)).wrapping_neg();
            }
            if (i+1)*64 > range.end {
                mask &= (1u64 << (range.end % 64)) - 1;
            }
            ret += (self.set[i] & mask).count_ones() as usize;
        }
        ret
    }

    /** Iterate over self */
    #[allow(dead_code)]
    pub fn iter(&self) -> BitSetIterator {
        BitSetIterator { set: &self, offset:0, cur:0 }
    }
}

pub struct BitSetIterator<'a> {
    set: &'a BitSet,
    offset: usize,
    cur: u64
}

impl <'a> Iterator for BitSetIterator<'a> {
    type Item = usize;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur != 0 {
                let ret = self.cur.trailing_zeros() as usize + (self.offset-1)*64;
                self.cur &= self.cur-1;
                debug_assert!(self.set.contains(ret));
                return Some(ret);
            } else if self.offset >= self.set.set.len() {
                return None;
            } else {
                self.cur = self.set.set[self.offset];
                self.offset += 1;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::tilematrix::bitset::BitSet;
    use rand::{Rng,thread_rng};

    #[test]
    fn test_bitset() {
        for _ in 0..100 {
            let start = thread_rng().gen_range(0..1000);
            let end = thread_rng().gen_range(start..1001);
            let set = BitSet::from_range(1000,start..end);
            assert_eq!(set.len(), end-start);
            assert_eq!(set.count_within(start..end), end-start);
            assert_eq!(set.count_within(0..1000), end-start);
            for i in 0..start {
                assert!(!set.contains(i));
            }
            for i in start..end {
                assert!(set.contains(i));
            }
            for i in end..1000 {
                assert!(!set.contains(i));
            }
        }
    }
}