
# Rust port

* Use Borrow to make the code more generic
* Is there a way to have a borrowed and owned version?  Eg maybe Cow?
* Use boxed slices instead of Vec?
* Mark vectorized APIs as unsafe.
* Clean up code TODOs
* Rename functions for clarity
* Decide on assert! / debug_assert! / nothing
* C / C++ interface
* Demo app
* File handling / serde

## Rust performance:

The Rust version has performance tradeoffs vs C.  Overall, the Rust
matrix solvers are faster for large matrices, especially on Intel,
and the row rearrangement (interleave_rows/partition_rows) is slower
but more cache-friendly.  This results in a slower uniform map solver
on M1 and a wash on Intel.

The Rust nonuniform map code is simpler and faster, so it is faster
everywhere.

Pseudoinverse is still unoptimized and should be improved.

The Rust code is single-threaded, and should eventually be parallelized
at least optionally.
    
# For release

* Detect vector acceleration instead of compiling it in fixed
* Stabilize interface and format

## Testing and documentation

* Test various things with zero allowed tries, zero items, zero value bits, etc.

## API / included features

* File handling
* Demo: CRL compression.

# Longer term

* Armv7 and x86 support
* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion; doesn't currently fit in memory)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Make production-quality.
