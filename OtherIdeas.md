# Other ideas tested

I explored several other options when building this library.  Here are some that
seem sufficiently non-proprietary to publish.

# Matrix structure

## Wider ribbon filters

We can get to O(1) additive overhead if the width of the ribbon is O(log n
sqrt(n)).  This is pretty slow though: it takes O(n^2) time to solve and
O(sqrt(n)) time to query.

## Frayed ribbons with more / less overhead

With constant overhead, the block size beta needs to grow faster: it needs to be
at least 64 for hundreds of millions of keys, vs 32 with an overhead of 0.1%.

With larger overhead, in principle construction should take quasilinear time.
But in practice, the solving algorithm is bottlenecked by data movement, so at
larger overhead it isn't competitive with ribbon filters. As a result, I fixed
the overhead at 0.1%.

## Setting the block size to 2 bytes instead of 4

This doesn't show any performance advantages, and starts causing failures at
around 100 million rows.

## Shaping the matrix to the solver ("Calderesque" matrices)

Instead of setting the difference between two blocks i-j to be small, we could
set the xor-difference to be small (more specifically, to usually have the MSB
be small, and the rest be random).  This would target the solving algorithm
directly, and result in better solving performance. Essentially, while |i-j|
small usually implies i^j small, it doesn't always, so some rows end up in large
blocks despite a small difference. This doesn't happen if we target the
xor-distance instead.

However, if the number of blocks isn't a power of 2, then the xor-distance must
be sampled in a somewhat more complex way, which hurts query speed. Furthermore,
this situation seems harder to analyze.  So I dropped the code which does that.

## Discretized offsets

Instead of allowing i-j to attain any value, it can be restricted to a power of
2 or 4 (or something else).  This can be sampled quickly: with probability p,
take i-j=1; and with probability (1-p), take i-j = LSB of random^2 (for powers
of 4), where LSB(x) = x &~ x-1.  This can be backstopped by or'ing with, eg, a
power-of-2 overhead factor like 1024. This might be easier or harder to analyze.

Experiments suggest a slightly higher solving probability for the same mean
distance, but it's trickier to tune p (eg around log(n)) and the solver isn't
any faster in the end.  Query speeds are also not improved.

However, the hierarchical solver spends a lot of time shuffling data around row
by row.  If the distances were discretized, there would be a smaller number of
possible row shapes, so submatrices might not need to be rearranged as often.

## Adding or splitting rows

When the matrix is overprovisioned, it might be possible to speed up solving by
adding rows with carefully-chosen block indices but random offsets.  This would
add more rows to the smaller matrices, while removing columns from the larger
matrices.  The offsets could be chosen in such a way that each collection of
columns remain row-deficient (by more than a certain margin, to ensure high
solution probability).

This seems worth doing if it can be done correctly, because the final matrices
solved may be 2-8x wider than they are tall.  Adding extra rows earlier would
thus save up to 2-8x performance on the slowest steps.

Similarly, we could split larger rows in half, where a row with two blocks of
((A at offset I) + (B at offset J)) * M = C could be replaced by

(A at offset I) * M = random

(B at offset J) * M = C - random

When overprovisioned by a fraction delta, the current system replaces the worst
delta-fraction of rows with ones with smaller offset.  This requires the query
algorithm to know delta, which is annoying -- except that this filter seems
mostly worth using with delta very small.  We could instead have the query
algorithm not know delta, but instead split the largest delta (or so) rows.

I tried this, and it seemed to perform worse than the existing method, and had
significantly more code to figure out which rows to split.  But maybe there's a
way to make it work.

## Forcing the blocks to be nonzero

If a row is sampled as entirely zero, then the entire matrix reduction always
fails.  So it might make sense to force the row to nonzero, or force both halves
to nonzero.  But this didn't seem to help in testing.

## Sharding

It's possible to shard filters in order to improve solve speed.  For something
like a matrix or xor-sat filter it's essential.  But it increases complexity,
space overhead and query cost, and it doesn't improve the solve speed by a huge
amount.  So at least for the motivating use case (CRL compression or other
static dictionaries compiled once and used many times) I don't think it's
worthwhile.

## Multi-threading

Multithreading is turned off by default.  It buys a factor of 2.5 or so with
4-8 threads.  The bottleneck is mainly solving the largest few matrices in the
center -- to remove it, we would need to multithread the matrix solver too,
which is basically doable but maybe beyond what I want to do for a research
implementation.  It might be possible to drop in M4RI instead of my solver to
remediate this.

To multithread non-uniform static functions, we would also need to parallelize
the hashing and querying steps that it performs.  This is totally possible, but
again kind of annoying for a research implementation.




# Library structure

## Using a struct instead of bytes

If you're serializing a function to a file to be transported over the internet,
then byte-serialized headers and bounds checks and such are required.  But if
you're just using one locally, then it makes more sense to organize the object
as a C struct with some headers and an array of words.  If the function was
compressed by a trusted source (e.g. the program itself or a step during its
compilation) then you wouldn't need sanity checks.

Maybe it would be better to split querying a static-function-as-bytes into two
steps: first parse the header into a struct and do all the bounds checks, and
then query using that struct. It would be cleaner and probably not much slower.
(TODO?)

## Making a header-only library, a la fastfilter_cpp

This would make lookups only slightly faster.  It might make integration easier
though, especially if you only need queries.

## Prehashed variant

The library uses a variant of the murmur3 hash function (with a wider seed), as
a balance of quality vs speed.  But maybe this balance doesn't make sense.  At
least the motivating use case is cryptographic data, and for that you probably
want to use SHA2(another salt, input) as a preprocessing step to make sure that
the data has no adversarial structure.  In that case, murmur3 is unnecessary,
and a simpler finalizer can be used instead.

Fastfilter_cpp assumes the input is prehashed and therefore reasonably random,
so it already does this.  Using a simpler finalizer ought to improve performance.

But in practice, it makes only a small impact on Intel, where queries seem to be
bottlenecked by the memory lookups which somehow don't seem to parallelize -- at
least I'm guessing that's why frayed ribbons are about twice as slow as standard
ribbons?

On Apple M1, frayed ribbon queries are about as fast as standard ribbon queries
(they didn't optimize for M1, but neither library is using intrinsics here I
think), and the murmur3 hash is a meaningful fraction of the timing.

But anyway, at least for the motivating use case, the whole thing is going to be
bottlenecked by SHA2/SHA3 anyway so this is pretty far into the bit-fiddling
territory.  And in other use cases, we may not be able to assume that the input
is prehashed.

## Use xxhash instead of murmur3

This is marginally faster, but I didn't want the extra dependency.




# Matrix solver

## Why didn't you use M4RI or gf2toolkit?

Mostly because they're licensed under the GPL, so linking them would make the
legal department nervous.  Also, writing tile_matrix.c was a lot of fun, and
uses a novel strategy.  And it's pretty fast: it goes toe-to-toe with M4RI for
multiplication and echelonizing up to 1000x1000 or so, and it's faster than M4RI
on small matrices.

It might still be worth substituting M4RI or gf2toolkit and benchmarking though:
I have no idea which is faster for just copying rows/columns around, and M4RI
supports multithreading.

## Different tile sizes

8x8 tiles seemed to perform best, in part because lookup instructions have 8-bit
output.  You might imagine that 16x16 would be better with AVX2's 256-bit
vectors, but the required permutations on 256-bit vectors are actually pretty
expensive, so it was slower and more complex in my tests.
