/*
 * @file lib.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Library main.
 */

/*!
Compressed mapping objects, also known as "static functions".

This crate provides a [`CompressedMap<K, V>`] object, which operates
somewhat like an immutable [`HashMap<K,V>`](std::collections::hash_map::HashMap):
it maps keys of type `K` to values of type `V`.  However, it has a significant
and intentional limitation: it does not store the keys to the map.  This
allows significant compression, but limits the use cases.

This crate also provides [`ApproxSet`] and a lower level [`CompressedRandomMap`].

# Use cases

[`CompressedMap`] and [`CompressedRandomMap`] are useful if you
want to compress a static map and distribute it to users, and you
know that the users will only use it for lookups on keys that are
in the map.

The motivating example is
[CRLite](https://blog.mozilla.org/security/2020/01/09/crlite-part-2-end-to-end-design/).
CRLite enables an aggregator to compress Certificate Revocation Lists (CRLs),
for end-users who only use it to check whether a given certificate is
revoked.  By collecting all certificates from participating CAs
in certain time period, using certificate transparency logs, the aggregator
knows all queries that the client might make.  It can then construct
(CRLite's version of) a [`CompressedMap<Cert,bool>`] mapping each
cert to whether it is revoked or not.  By discarding the keys to the
map and retaining only the key-value mapping, CRLite can compress them
extremely effectively.

This crate's `CompressedMap` is about 40% smaller than the ones in CRLite.
For a CRL in which 1% of certs are revoked, it uses about 1.1 bytes per
revoked cert.

There may be other use cases (malware databases? chess endgames??), but
it's important to keep in mind that, when querying, you can't tell whether
a key is in the map, and therefore whether you're getting garbage out.

[`ApproxSet`] is more generally useful.  It operates much like static
[Bloom filters](https://en.wikipedia.org/wiki/Bloom_filter), but it takes
about 30% less space.
 
# Compressed maps
 
Compressed maps can be constructed from a standard map `map` of one of the
usual mapping types such as [`HashMap<K,V>`](std::collections::HashMap) or
[`BTreeMap<K,V>`](std::collections::BTreeMap).  On lookup
of a `key` in the [`CompressedMap`], if `map[key] == value`, then `value` will
be returned.  But if `key` was not in `map`, then the `CompressedMap` has no
way to detect this, because it has discarded the keys.  It will instead return
some value that was in the map, arbitrarily.  (As a result, you cannot
construct a `CompressedMap` from an empty map.)
 
This compressed map implementation is most efficient for maps containing
hundreds to hundred-millions of keys, but only a few values.

## Code example

This toy example shows the compression that [`CompressedMap<u64,bool>`] provides
compared to a serialized [`HashMap`](std::collections::HashMap).

```
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
println!("hashmap={} bytes, cmap={} bytes, ratio={:0.1}",
    hash_sersize, sersize, hash_sersize as f64/sersize as f64);
println!("Shannon limit for {:0.2}%={} bytes, overhead={:0.2}%",
    p*100., (shannon/8.) as u64, (sersize as f64*8. / shannon - 1.) * 100.);
    
// Example output:
// hashmap=900008 bytes, cmap=3952 bytes, ratio=227.7, 0.32 bits/key
// Shannon limit for 5.03%=3596 bytes, overhead=9.87%
```

# Random compressed maps

A lower-level building block is [`CompressedRandomMap<K,V>`].  These work much
the same as `CompressedMap`s, but for `V` the support only values `V` which are
integers, or otherwise [`Into<u64>`](std::convert::Into) and
[`TryFrom<u64>`](std::convert::TryFrom).  They
are efficient when the values are approximately uniformly random up to a certain
bit length (e.g., when they are random in `0..16`).  When queried with a key
that wasn't in the original map, they return an arbitrary value of the appropriate
bit length --- not necessarily one of the original keys.
 
## Approximate sets

[`ApproxSet`]s are constructed from a set `set`, e.g. a [`HashSet`](std::collections::hash_set::HashSet).
When you query `approx_set.probably_contains(x)`, then if `set.contains(x)` you will 
always receive `true`.  On the other hand, if `!set.contains(x)`, then you will usually receive
`false`, but there is a false positive probability.  By default this is 2<sup>-8</sup>, but
you can control it using the `bits_per_value` field in [`BuildOptions`].

# Performance
## Space usage

[`ApproxSet`] and [`CompressedRandomMap`] use approximately `bits_per_value` bits per entry
in the map.  By default, this is 8 bits for [`ApproxSet`], and the maximum bit-length of
any value in the map for [`CompressedRandomMap`].

[`CompressedMap`] uses approximately H<sub>0</sub>(V) * (1+overhead) bits per entry.
Here H<sub>0</sub>(V) is
the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the
distribution of values --- e.g. it is 1 for a map whose values
are (50% `true`, 50% `false`), and 0.081 for a map that's (1% `true`, 99% `false`).  The
overhead is 0.001 to 0.11 (i.e. 0.1% to 11%), depending on the distribution: it's worst for maps
that split about 80%/20%.
 
[`CompressedMap`] also stores a table of possible values; internally it is compressing an
index into that table.  This value table is not compressed, so maps with many giant objects as
values won't compress very well.

All of these structures have small constant or nearly-constant overheads --
for example, they contain lengths, hash keys and padding.

## Time and memory

Querying any of these maps or sets is very fast, typically around 100-200 cycles
if the map is in cache.  The process typically uses 2 sequential groups of memory
lookups in a large array, and the memory lookups themselves are often nearby, so it should
be reasonably fast even if the map is on disk.  (TODO: support `mmap`.)

The construction part of this library is optimized for either AArch64 machines
(assumed to have NEON), or x86_64 machines with AVX2.  On other architectures,
lookups are still fast but set construction is very slow.  (TODO...)

Even with a supported vector unit, performance can be a bottleneck.  On an Apple M1, building
a [`CompressedRandomMap`] with 100 million entries takes around 2 minutes and 8 gigabytes of memory.
Construction is theoretically Õ(n<sup>3/2</sup>), but in practice is dominated by Õ(n) memory
copies and O(n log n) memory consumption.

Building an [`ApproxSet`] is similar in performance to a [`CompressedRandomMap`] of the
same size.
 
When building a [`CompressedMap`], if one value is vastly more populous than the rest,
it will usually not be used in most of the internal `CompressedRandomMap` objects.  As a
result, the building performance depends more on the population of the non-dominant values.
So for example, if 99% of the values are `false` and 1% are `true`, then building a map takes
around an order of magnitude less time and memory, and the largest [`CompressedMap`]
contains only 2-3% of the total population.

## Threading

With the `threading` feature enabled, building the core map objects can be multi-threaded.
Currently not all steps are threaded, but a 2-3x speedup can be expected for large maps. 
Smaller maps (< 100,000 entries) take longer to build with multi-threading because of the
synchronization overhead.

# Failure

Building a map is probabilistic, and will be retried a certain number of times before it
fails.  Each try succeeds approximately 90% of the time or more.  You can control the number
of tries in [`BuildOptions`]; with the default options a failure is negligibly unlikely in
most cases.  There are some caveats to this though:
 
* The failure analysis is heuristic, and might be wrong.  Likewise the code might be buggy.
* Maps with an enormous number of keys, e.g. more than about 128 billion for a [`CompressedRandomMap`],
  will always fail.
* Because [`CompressedMap`] needs a default value to return, trying to construct one from an empty
  map will always fail.
* None of these types are tolerant of duplicate keys (TODO?).  You should construct them
  from a type that you know does not contain duplicates, such as a `HashSet` rather than a `Vec`.
  The constructor does not detect this case, and will therefore try and fail to construct the map
  `max_tries` times, which may be extremely slow.  Since the keys are hashed, this also applies
  to keys that have identical hashes with SipHash128(1,3), e.g. if your `Hash` implementation
  has deterministic collisions for unequal keys.
* If you supply a key that's constructed non-randomly, if an attacker can predict it, then they
  can cause your map construction to repeatedly fail.
* If threading is enabled, a thread running out of memory may be treated as a failure, causing
  a slow and resource-intensive series of retries (TODO: support errors as well as options).

# Ownership

In order to facilitate referencing non-owned data (e.g. in an `mmap`'ed region), 
[`CompressedMap`], [`ApproxSet`] and [`CompressedRandomMap`] have lifetime parameters
for internal references.  They can instead use vectors, and do use vectors when built
using `build` methods, which gives them arbitrary lifetime.  You can change the
references to vectors by calling the `take_ownership` method.

# Serialization

Note! Until v0.2.0, the compression format is not finalized.

Serialization of maps is provided using the [`bincode`] crate.  All of [`CompressedMap`],
[`ApproxSet`], [`CompressedRandomMap`] implement
[`Encode`](bincode::enc::Encode) and [`BorrowDecode`](bincode::de::BorrowDecode).  Once
you `BorrowDecode` the object, you can take extend the lifetime with `take_ownership`.
Because `bincode` doesn't provide a serialized size anymore, this crate exports
[`serialized_size`] to give the size of an [`Encode`](bincode::enc::Encode) object.

For compatibility, simplicity and speed, please use the [`STD_BINCODE_CONFIG`] when
calling `bincode`'s serializers.  That way all the u32's won't get recode in varint format.

Because the maps and sets in this crate do not represent their keys, this doesn't rely
on `K:Encode` or `K:Decode` or `Clone`.  Because [`CompressedMap<K,V>`] can take
nearly arbitrary values, and represents those value, it does rely on these traits for
`V`.
 
# Internals

Internally, [`CompressedRandomMap`] is implemented as a "frayed ribbon filter".  This is
a matrix `M`, such that `F(k) = M * encode(hash(k)) ^ hash(k)` for each key `k`
in the map.   The encoding of hashes into vectors chooses two 32-bit blocks with random
contents, according to a distribution that usually places them near to each other;
within the blocks the vector is pseudorandom (according to the hash), and elsewhere
it is zero.  This is called a frayed ribbon filter because it's similar to
[ribbon filters](https://engineering.fb.com/2021/07/09/data-infrastructure/ribbon-filter/).

The building process involves solving the linear system constrained by
`M * encode(hash(k)) ^ hash(k) = v` for all `(k,v)` in the map.  This can be done
efficiently in a hierarchical pattern.

An [`ApproxSet`] simply maps `F(x) = 0` for all `x` in the set.  Since each `x` gets
a pseudorandom offset, the false positive probability is 2<sup>`-bits_per_value`</sup>;
without that offset, there would be a danger that e.g. the all-zeros kernel vector might
be chosen, which would have a false positive probability of 1.

A [`CompressedMap`] is implemented by choosing an `i32` "Locator" according to
several [`CompressedRandomMap`]s.  Each possible output is assigned a naturally aligned
interval in `0..2`<sup>`32`</sup>, and if the locator is in that interval that value is the output.
The individual maps are generated including only the `(key, value)` pairs that are required
for correctness.
*/


/**
 * GF(2) matrix ops (internal; exposed for bench)
 *
 * GF(2) matrix ops implemented using tiles and the Method of the
 * Four Russians.  This is internal, and only exposed for benchmarking.
 */
#[cfg(bench)]
pub mod tilematrix;

#[cfg(not(bench))]
mod tilematrix;

mod uniform;
mod nonuniform;

#[cfg(feature="cffi")]
mod cffi;

mod size;

pub use uniform::{BuildOptions,CompressedRandomMap,ApproxSet,STD_BINCODE_CONFIG,KeyedHasher128,DefaultHasher};
pub use nonuniform::{CompressedMap};
pub use size::{serialized_size};

