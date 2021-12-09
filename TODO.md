# For release

## Correctness / suitability

* FIXME BUG: nonuniform maps don't work if an item has zero population.
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
