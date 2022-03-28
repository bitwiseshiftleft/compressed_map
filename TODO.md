
# Rust port

* Since we aren't counting outliers, no need to backtrack in nonu::build; can just do for 0..nphases
* Clean up code TODOs
* Rename functions for clarity
* Rename Map and NonUniformMap to CompressedMap etc.
* Decide on assert! / debug_assert! / nothing
* C / C++ interface
* Demo app
* File handling / serde
* More documentation

## Rust performance:

The Rust version is slower than the C version except for large matrices.
Internally, the matrix solvers are faster than C for large matrices (due
mainly to larger tiles) but rearranging rows is slower (but more cache-
friendly).

Hot spots in the profile:

* interleave_rows (add BMI2 version?)
* partition_rows  (can it be further improved?)
* pseudoinverse: the current code is optimized for small tiles.

We could try the C strategy to reduce rearrangements?  I tried this in branch
cstrat.  It's faster for small n, especially on M1.  However for large n, 
especially on Intel, it isn't.  Maybe it's harder on the CPU cache due to
the larger number of random row writes?

Also the current Rust code has no threading.
    
# For release

* Detect vector acceleration instead of compiling it in fixed
* Stabilize interface and format

## Testing and documentation

* Test various things with zero allowed tries, zero items, zero value bits, etc.
* Check the rust docs

## API / included features

* File handling
* Demo: CRL compression.

## Cleanliness

* Make code more portable -- probably not full cmake though.
* Make tile_matrix_t a one-element array too?

## Performance

* Take better advantage of items being sorted to natural alignment while building?

# Longer term

* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion; doesn't currently fit in memory)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Make production-quality.
