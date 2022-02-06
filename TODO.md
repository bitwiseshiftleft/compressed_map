# For release

## Correctness / suitability

* FIXME BUG: nonuniform maps don't work if an item has zero population.
* FIXME BUG: nonuniform maps rarely, but non-negligibly, fail to build.
* Implement the simpler solution for formulate_plan.
* Make build use a hash table to store item pointers, instead of an array, for sparse items.

## Testing and documentation

* Test the C++ interface better.
* Make the test/bench frameworks more consistent.
* Test various things with zero allowed tries, zero items, zero value bits, etc.
* Doxyfile

## API / included features

* File-handling / ser / deser API.
* Demo: CRL compression.

## Cleanliness

* Make code more portable -- probably not full cmake though.
* Make tile_matrix_t a one-element array too?

## Performance

* Sort items to natural alignment when building.  Put balance last (simplifies deser I guess)
* Use sorting to optimize building multi-response maps, since we don't have to query (except for the balance class, darn)
* Is (2nd highest, highest, then lowest) the right order on queries?  Maybe it should just be highest .. lowest?

## Builder / deduping hash table

* Speed up builder hash table impl?
* Should we just forget murmur3/siphash and go with SHA256, hoping for HW acceleration?
* Test builder hash table impl (using C++?)
* More tests around builder expanding memory.

# Longer term

* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Comprehensive unit tests.
* Make production-quality.

## Optimizing for many different results

For tables with many different possible results, queries are somewhat slow because many uniform maps might need to be consulted.  However, it may be possible to reduce this at a small cost in size efficiency.

One part of the approach is to properly align the intervals, and optimize cases where some intervals need not be considered in a given phase.

However, there is a possible approach where fewer tables would need to be consulted even in the worst case.  Arrange all intervals so that they are power-of-2 sized and aligned, except for one
case (the "default" case).  Then the phases could be defined by the pattern of bits in the default interval size.  For example, if the encoding table is:

* 00x -> A
* 010 -> B
* 1xx or x11 -> C

This would normally require 3 leveles.  But we can randomly permute the last two bits (according to the hash function) to get:

* 0x0 // 00x -> A depending on hash
* 010 // 001 -> B depending on hash
* 1xx or x11 -> C

Then with about sqrt(n) extra space, the calculation can be adjusted so that the least significant two levels are the same size (have about the same number of A's in them), so that the table collapses to only 2 levels.  It might be further possible to make some of the A's shared between the two tables (using xor instead of a rotation), but that might extend solve time too much.