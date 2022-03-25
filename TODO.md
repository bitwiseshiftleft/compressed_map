
# Rust port

* Nonuniform maps
* Approximate sets
* Clean up code TODOs
* Rename functions for clarity
* Each pass of set building should just take an iterator.
* C / C++ interface
* Demo app
* File handling / serde

## Rust performance:

The Rust version is slower than the C version except for large matrices.
Internally, the matrix solvers are faster than C (due mainly to larger tiles)
but rearranging rows is slower.

Hot spots in the profile:

* interleave_rows
* partition_rows
* pseudoinverse

We could try the C strategy to reduce rearrangements?

Also Rust has no threading.
    
# For release

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
