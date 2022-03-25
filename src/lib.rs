
/**
 * GF(2) matrix ops (internal)
 *
 * GF(2) matrix ops implemented using tiles and the Method of the
 * Four Russians.  This is internal, and only exposed for benchmarking.
 *
 * TODO: find a way to make it private.
 */
pub mod tilematrix;

/**
 * Uniform static functions and approximate sets.
 *
 * Uniform static functions compress dictionaries `K -> u64`, where `K:Hash`.
 * You can look items up in the dictionary, but you can't list the keys.
 * If you query something not in the dictionary, you'll get an arbitrary
 * answer.
 *
 * These functions are "uniform" in that they don't take advantage of any
 * possible skewed distribution of the values, other than the number of bits
 * in the longest value.  For that, use the nonuniform module.
 *
 * This module also implements approximate sets, which are similar to Bloom
 * filters.  You can build an approximate set `s` from a collection `c` of items.
 * You can then query `s.probably_contains(x)`.  If `x` was in `c`, then tihs always
 * returns yes.  Otherwise, it usually returns no, but there is a false
 * positive probability which is about `2^-b`, where `b` is the number of bits
 * used per element of `c`.
 */
pub mod uniform;

use tilematrix::tile;
