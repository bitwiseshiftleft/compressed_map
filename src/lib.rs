/*!
 * Compressed mapping objects.
 * 
 * This crate provides a [`CompressedMap<K, V>`] object, which operates
 * somewhat like an immutable [`HashMap`](std::collections::hash_map::HashMap)
 * from keys of type `K` to values of type `V`.  However, it has a significant
 * and intentional limitation: it does not store the keys to the map.  This
 * provides better compression, but limits the use cases.
 * 
 * This crate also provides [`ApproxSet`] and a lower level [`CompressedRandomMap`].
 * 
 * # Compressed maps
 * 
 * Compressed maps can be constructed from a standard map `map` of one of the
 * usual mapping types such as `HashMap<K,V>` or `BTreeMap<K,V>`.  On lookup
 * of a `key` in the [`CompressedMap`], if `map[key] = value`, then `value` will
 * be returned.  But if `key` was not in `map`, then the CompressedMap has no
 * way to detect this, because it has discarded the keys.  It will instead return
 * some value that was in the map, arbitrarily.  (As a result, you cannot
 * construct a `CompressedMap` from an empty map.)
 * 
 * This compressed map implementation is most efficient for maps containing
 * hundreds to hundred-millions of keys, but only a few values.  The motivating
 * example is certificate revocation, à la
 * [CRLite](https://blog.mozilla.org/security/2020/01/09/crlite-part-2-end-to-end-design/).
 * In this example, a `CompressedMap<Certificate,bool>` could represent which
 * certificates are revoked and which are still valid.
 * 
 * # Random compressed maps
 * 
 * A lower-level building block is [`CompressedRandomMap<K,V>`].  These work much
 * the same as `CompressedMap`s, but they return integer types such as `i8`.  They
 * are efficient when the values are approximately uniformly random up to a certain
 * bit length (e.g., when they are random in `0..32`).  When queried with a key
 * that wasn't in the original map, they return an arbitrary value of the appropriate
 * bit length --- not necessarily one of the original keys.
 * 
 * # Approximate sets
 * 
 * These operate much like static [Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter).
 * They are constructed from a set `set`, e.g. a [`HashMap`](std::collections::hash_set::HashSet).
 * When you query `approx_set.probably_contains(x)`, then if `set.contains(x)` you will 
 * always receive `true`.  On the other hand, if `!set.contains(x)`, then you will usually receive
 * `false`, but there is a false positive probability.  By default this is 2<sup>-8</sup>, but
 * you can control it using the `bits_per_value` field in [`BuildOptions`].
 * 
 * # Space usage
 * 
 * [`ApproxSet`] and [`CompressedRandomMap`] use approximately `bits_per_value` bits per entry
 * in the map.  By default, this is 8 bits for [`ApproxSet`], and the maximum bit-length of
 * any input for [`CompressedRandomMap`].
 * 
 * [`CompressedMap`] uses approximately H0(V) * (1+overhead) bits per entry.  Here H0(V) is
 * the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the
 * distribution of values --- e.g. it is 1 for a map whose values
 * are (50% `true`, 50% `false`), and 0.081 for a map that's (1% `true`, 99% `false`).  The
 * overhead is 0.001 to 0.11 = 0.1% to 11%, depending on the distribution: it's worst for maps
 * that are about 80% one value and 20% another value.
 * 
 * [`CompressedMap`] also stores a table of possible values; internally it is compressing an
 * index into that table.  This table is not compressed, so maps with many giant objects as
 * values won't compress very well.
 * 
 * All of these structures have small constant or nearly-constant overheads as well --
 * for example, they contain lengths, hash keys and padding.
 * 
 * # Performance
 * 
 * Querying any of these maps or sets is very fast, in most cases around 100-200 cycles
 * if the map is in cache.  The process typically uses 2 sequential groups of memory
 * lookups in a large array, and the memory lookups themselves are often nearby, so it should
 * be reasonably fast even if the map is on disk.  (TODO: support `mmap`.)
 * 
 * The construction part of this library is optimized for either AArch64 machines
 * (assumed to have NEON), or x86_64 machines with AVX2.  On other architectures,
 * lookups are still fast but set construction is very slow.  (TODO...)
 * 
 * Even with a supported vector unit, performance can be a bottleneck.  On an Apple M1, building
 * a [`CompressedRandomMap`] with 100 million entries takes around 2 minutes and 8 gigabytes of memory.
 * Construction is theoretically Õ(n<sup>3/2</sup>), but in practice is dominated by Õ(n) memory
 * copies and O(n log n) memory consumption.
 * 
 * Building an [`ApproxSet`] is similar in performance to a [`CompressedRandomMap`] of the
 * same size.
 * 
 * The performance of building a [`CompressedMap`] depends mostly on the population of its
 * second-most-common value; the most-common values usually just have to be counted and queried
 * against the map, and only a minority of them are used in the expensive part of the solver.
 * So for example, if 99% of the values are `false` and 1% are `true`, then building a map takes
 * around an order of magnitude less time and memory (not two orders of magnitude because the
 * values still need to be stored, queried etc).
 * 
 * TODO: building the maps can be multithreaded without too much difficulty.  It won't
 * parallelize perfectly unless the matrix solver can be parallelized as well, but it should
 * achieve a 2-4x speedup.
 * 
 * # Failure
 * 
 * Building a map is probabilistic, and will be retried a certain number of times before it
 * fails.  Each try succeeds approximately 90% of the time or more.  You can control the number
 * of tries in [`BuildOptions`]; with the default options a failure is negligibly unlikely in
 * most cases.  There are some caveats to this though:
 * 
 * * The failure analysis is heuristic, and might be wrong.  Likewise the code might be buggy.
 * * Maps with an enormous number of keys, e.g. more than about 128 billion for a [`CompressedRandomMap`],
 *   will always fail.
 * * Because [`CompressedMap`] needs a default value to return, trying to construct one from an empty
 *   map will always fail.
 * * None of these types are tolerant of duplicate keys (TODO?).  You should construct them
 *   from a type that you know does not contain duplicates, such as a `HashSet` rather than a `Vec`.
 *   The constructor does not detect this case, and will therefore try and fail to construct the map
 *   `max_tries` times, which may be extremely slow.  Since the keys are hashed, this also applies
 *   to keys that have identical hashes with SipHash128(1,3), e.g. if your `Hash` implementation
 *   has deterministic collisions for unequal keys.
 * * If you supply a key that's constructed non-randomly, if an attacker can predict it, then they
 *   can cause your map construction to repeatedly fail.
 */


/**
 * GF(2) matrix ops (internal; exposed for bench)
 *
 * GF(2) matrix ops implemented using tiles and the Method of the
 * Four Russians.  This is internal, and only exposed for benchmarking.
 *
 * TODO: eventually this should be private
 */
pub mod tilematrix;
mod uniform;
mod nonuniform;

pub use uniform::{BuildOptions,CompressedRandomMap,ApproxSet};
pub use nonuniform::{CompressedMap};

use tilematrix::tile;
