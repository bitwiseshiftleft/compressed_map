
# Rust port

* Nonuniform maps
* Clean up code TODOs
* Rename functions for clarity
* C / C++ interface
* Demo app
* File handling / serde
* More documentation

## Rust performance:

The Rust version is slower than the C version except for large matrices.
Internally, the matrix solvers are faster than C (due mainly to larger tiles)
but rearranging rows is slower.

Hot spots in the profile (TODO: update since partition_rows was removed):
* interleave_rows
* pseudoinverse

Also Rust has no threading.
    
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
