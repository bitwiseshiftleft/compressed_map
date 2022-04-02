# Ser/deser

* Check that bincode code looks sane
* Stabilize interface and format

# Other release items

* Convenience methods, at least, for file handling
* Enable mmap?
* C / C++ interface / dynamic lib
* Demo app
* Examples in doc
* Make sure it builds automatically with AVX2 when possible
* Distinguish between "out of memory", "can't create thread" etc, and "matrix is not invertible"
* Deal with overflow cases with billions of items in nonuniform maps, where 0 rounds up to 1.

# Performance

* Multithread hashing even if we aren't multithreading bucketsort.  (Using Rayon??)
* Improve optimization of the threaded version
* Improve pseudoinverse
* More profile-driven optimization
* Compare which parts are still faster in C

## API / included features

* File handling
* Demo: CRL compression.

# Longer term

* no_std core for embedded systems
* Armv7 and x86 support, with NEON / AVX2
* Better interface for tile matrices
* Complete multi-threading impl
* Test on very large data sets (eg 1 billion; doesn't currently fit in memory)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Make production-quality.
