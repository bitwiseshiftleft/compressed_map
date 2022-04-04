# Compressed maps without the keys, based on frayed ribbon cascades

This research-grade library aims to implement space-efficient static functions, with a performance profile suitable for evolutions of [CRLite](https://blog.mozilla.org/security/2020/01/09/crlite-part-2-end-to-end-design/).  Of course, it's probably applicable to other domains.

Written by Mike Hamburg.  © 2020-2022 Rambus Inc, MIT licensed.  This attic_c code incorporates public-domain SipHash code by Jean-Philippe Aumasson
and Daniel J. Bernstein; and code based on public-domain Murmur3 code by Peter Scott.

This is a research-grade implementation.  Don't deploy it in production.

## Static functions, aka CompressedMap

The static function problem is to compress a map `M` of type `K -> V`.
The compression is allowed to discard the keys. With the compressed map,
it should be possible to determine `M[k]` for all keys `k` of `M`.
However, it might not be possible to list all the map keys, nor to
determine whether some key is in the map or not. If you query on
some `k` not in `M`, then an arbitrary result will be returned.

This library provides `CompressedMap`, which is a static function from
hashable types `K` to values `V`.  The values are stored in the structure,
so they should be clonable for use within a program, or serializable if
you want to serialize it.  The `CompressedMap` format is optimized for
cases where the values are few and small, e.g. booleans, small integers,
short strings etc.  The best use case is for exactly two values (eg `true`,
`false`) and millions or more keys.  It does not attempt to compress the
values, so don't use this structure to store large strings.

`CompressedMap` will collect statistics on the distribution of the values
and use this information to minimize storage size.  It uses slightly more
storage space per key than the Shannon entropy of the values: asymptotically
between 0.1% more and 11% more.  This is typically about 40% less space
than CRLite.

## Compressed random maps

This library also provides `CompressedRandomMap`.  This maps hashable
types `K` to integer types `V` of at most 64 bits.  Unlike a `CompressedMap`,
a `CompressedRandomMap` doesn't store its values, and doesn't use
statistics to try to reduce the storage size: it instead works well
when the value is a uniform random value.

## Approximate sets

The closely-related approximate set problem is to compress a set `S` of
objects.  You can query an element `x` to determine whether it's in
`S`.  If it really is in `S`, then the query will always return `true`.
If not, then it will usually return `false`, but there will be false
positives which instead return `true`.

This library provides `ApproxSet`, which is an approximate set of hashable
keys.  Compared to Bloom filters, `ApproxSet`s take several times longer
(and several times more memory) to construct; are faster to query; and
take about 30% less space.

## Other details

See the Rust documentation for construction, serialization, caveats etc of
these structures.

# Internals

The library uses "frayed ribbon filters", so called because they are similar
to [ribbon filters](https://engineering.fb.com/2021/07/09/data-infrastructure/ribbon-filter/).
These take asymptotically Õ(n^(3/2)) time and Õ(n) memory to construct.
In practice, they top out at a few hundred million to a billion rows on
commodity hardware, due to the memory consumption, unless someone can come
up with a lighter building algorithm.

For a `CompressedMap`, not all the keys are used in each level.  Roughly,
if there is a single "dominant" value that appears vastly more often than
the rest, then the memory consumption depends on some multiple (2-3x) of
how often the non-dominant values appear.