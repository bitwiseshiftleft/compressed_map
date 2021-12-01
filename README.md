# Frayed Cascade: static functions tuned for CRL compression

This research-grade library aims to implement space-efficient static functions, with a performance profile suitable for evolutions of [CRLite](https://blog.mozilla.org/security/2020/01/09/crlite-part-2-end-to-end-design/).  Of course, it's probably applicable to other domains.

Written by Mike Hamburg.  © 2020-2021 Rambus Inc, MIT licensed. 

This is a research-grade implementation.  Don't deploy it in production.  Also I'm just a cryptographer moonlighting as a data structures researcher.  It's entirely likely that someone will make a better filter next year.

## Static functions

The "static function problem" or "retrieval problem" is essentially to compress a dictionary *D*: *K* &rarr; *V*.  From the compressed form, you must be able to find *D*\[*k*\] for any key *k* in the dictionary's key set.  Here *V* should be a small, fixed-sized codomain, such as bits or bytes.

If you look up a key *k* that isn't in the key set *K*, the static function doesn't typically raise an error.  Instead it's allowed to return an arbitrary result.  This enables better compression: we don't have to encode the dictionary's key set, but only its key-to-value mapping.  In libfrayed, looking up a key that's not in *K* returns a (non-cryptographically) pseudorandom result.

## Approximate set membership

It's possible to use a static function for approximate set membership, replacing a [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter).  This encodes a set *S* in a static function with an *v*-bit value space, by compressing the dictionary *D*\[*k*\] := 0 for each *k* in *S*.[^1]  From the compressed representation, you can check whether *k* is in *S*: if *k*&in;*S* then you will always get *D*\[*k*\] == 0.  In the case that *k*&notin;*S* then this holds for only a 2^-*m* fraction of the keys.  Libfrayed is suitable for approximate set membership, but Facebook's [ribbon filters](https://engineering.fb.com/2021/07/09/data-infrastructure/ribbon-filter/) are typically faster.

[^1]: In general, you would set D\[*k*\] := hash(*k*).  But libfrayed already xors a hash(*k*) term into the output, so this isn't necessary.

# CRLite's use case

This library includes two implementations of static functions, inspired by the problem of compressing certificate revocation lists, à la [CRLite](https://blog.mozilla.org/security/2020/01/09/crlite-part-2-end-to-end-design/).  With CRLite, a certificate authority or third-party aggregator (e.g. Mozilla) can compress a dictionary that maps unexpired certificates to their revocation status (revoked or still valid).  This uses much less space, and is faster to query, than a conventional CRL.

This gave a list of design goals:

* We want to compress a dictionary *D*: *K* &rarr; *V*, with hundreds of millions of keys but only a few distinct values.  Typically there are only two values: "still valid" and "revoked", but you might want to include a reason for revocation, and libfrayed supports this.
* The keys of the dictionary are serial numbers or hashes of certs, and are assumed to be uniformly random.
* Typically 99% of certs are not revoked, so the values of the dictionary are assumed to be i.i.d. non-uniformly random: typically they are 1 with small probability *p* and 0 with probability *1-p*.
* The compressed form must be small.  The Shannon limit |*K*| (*p* lg *p* + (1-*p*) lg (1-*p*)) gives a lower bound on the size that can be achieved.  CRLite achieves 1.44-3&times; the limit.  We would like to do better.
* The implementation's performance should still be acceptable if *p* is in fact large, e.g.\ if there is another Heartbleed and half the certs on the Internet must be revoked.  The resulting compressed CRL can't be as small if *p* is near 1/2, because of the Shannon limit.
* Queries must be very fast.
* Compressing the dictionary is done once hourly or daily.  It can take seconds or minutes with hundreds of millions of keys, but it shouldn't take hours.

Libfrayed achieves a compression ratio within about 0.1%-11% of the Shannon lower bound, in expecation, plus a few dozen bytes of metadata.  The ratio depends on *p*: the 11% case is when 20% of certs are revoked.  If very few are revoked, or if about half are revoked, then it's closer to Shannon.  Compression is fast-ish, taking a few microseconds per revoked cert, plus a fraction of a microsecond per non-revoked cert, on one core.  Queries take well under a microsecond if the dictionary is in RAM, and use only a few random lookups in case it isn't.

## No support for incremental updates

We also wanted to support efficient incremental updates, so that each day a small patch could be applied to an existing CRL to keep it up-to-date.  The approach used here doesn't achieve that.  But the daily updates can just be a CRL of all certs revoked that day.  This gives fast compression and nearly optimal bandwidth usage.  The performance is worse, because each received cert must be queried against every CRL since it was issued.  For a 90-day cert this might be 90 queries, which takes well under a millisecond if the CRLs are in RAM, but it's slower if they're on a slow drive.  This solution works but it's inelegant.  It also uses more than the optimal amount of space on drive.

# Frayed ribbon filters

The higher-level static functions are built by using an encoding technique on top of a lower-level static function.  The lower-level function assumes that the values in the dictionary are uniformly random: it still works for non-uniform values, but it doesn't get better compression by taking advantage of the non-uniformity.

## Comparison to ribbon filters

Libfrayed implements its uniform static functions using _frayed ribbon filters_, which are similar to [ribbon filters](https://engineering.fb.com/2021/07/09/data-infrastructure/ribbon-filter/)[^2].    Ribbon filters can implement either approximate set membership or general static functions, but they're optimized for set memebrship and have more space overhead for the general case.

[^2]: I invented frayed ribbons independently, but Facebook published first so they got the naming rights.

By constrast, frayed ribbon filters are slower, with compression taking around an order of magnitude longer.  But they're suitable for general static functions, and have very low space overhead.  This library is tuned for 0.1% overhead.  The other ~10.9% comes from the encoding that handles non-uniform values.

## Implementation details

Frayed ribbons are different from ribbon filters in two major ways.  First, they align the column support into 32-bit blocks.  This simplifies everything and improves query performance when the codomain is only a few bits rather than e.g.\ a whole byte.  (I guess that makes our ribbons [pink](https://en.wikipedia.org/wiki/Pinking_shears)?)

Second, instead of the ribbon's support being contiguous, it consists of two blocks with a distance of O(log n) / random^2: they are often contiguous, but may instead be separated by a usually-small distance.  An alternative, hierarchical solution strategy takes O(n^(3/2) log n) time to solve such a matrix.  Experimentally, the exponent seems to be smaller than 3/2, more in the neighborhood of 1.3.  Or maybe the leading exponent really is 3/2, but there's also a large linear term from row copying or something.

I don't have a proof that such matrices are soluble with high probability.  But I tested it exhaustively with 100 random trials for row sizes up to 1 million.  The typical result was 96-97%, with the lowest outlier at 85%.  I also tested a sparser set up to 100 million rows, with the same typical success rate.

### Tile matrix solver

The matrix solver is implemented on top of a generic dense-matrix solving library.  This library uses a "method of the four Russians" approach, but based on vector permutations instead of large tables.  This approach may be novel.  The idea is to divide the matrices into 8x8 tiles.  When multiplying a tile T by a row of tiles R, the solver caches values T*N for each nibble N, and uses vectorized table instructions (in AVX2 or NEON) to multiply quickly by the whole row.

This approach doesn't get the full log n speedup of the four Russians method, but it's simple and fast.  In particular, the libfrayed solver seems to be faster than [M4RI](https://github.com/malb/m4ri) for small matrices but slower for large ones, with a crossover point of around 1000x1000.  I'm not sure if M4RI would be faster, but it's GPL-licensed and I wanted to release under MIT, so I wrote my own.

### Notes
