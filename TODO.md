
# Rust port

* Use outer struct with AsRef, and eliminate MapCore?
* Mark vectorized APIs as unsafe.
* Decide on assert! / debug_assert! / nothing
* C / C++ interface
* Demo app
* Examples in doc
* Convenienc methods for file handling
* Make sure it builds with AVX2 when possible
* Distinguish between "out of memory", "can't create thread" etc, and "matrix is not invertible"
* Deal with overflow cases with billions of items in nonuniform maps.
* Multithread hashing even if bucketsort isn't multithreaded.

## Rust performance:

The Rust version has performance tradeoffs vs C.  Overall, the Rust
matrix solvers are faster for large matrices, especially on Intel,
and the row rearrangement (interleave_rows/partition_rows) is slower
but more cache-friendly.  This results in a slower uniform map solver
on M1 and a wash on Intel.

The Rust nonuniform map code is simpler and faster, so it is faster
everywhere.

The threaded version still isn't very optimized.

Pseudoinverse is still unoptimized and should be improved.
    
# For release

* Detect vector acceleration instead of compiling it in fixed
* Stabilize interface and format

## Testing and documentation

* Test various things with zero allowed tries, zero items, zero value bits, etc.

## API / included features

* File handling
* Demo: CRL compression.

# Longer term

* no_std core for embedded systems
* Armv7 and x86 support
* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion; doesn't currently fit in memory)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Make production-quality.
