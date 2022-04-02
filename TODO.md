# Other release items

* Convenience methods, at least, for file handling
* Enable mmap?
* C / C++ interface / dynamic lib
* File handling
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

# Longer term

* Make production-quality (1.0).
* no_std core for embedded systems
* Armv7 and x86 support, with NEON / AVX2
* Better interface for tile matrices; release as its own crate?
* Test on very large data sets (eg CompressedRandomMap with 1 billion entries; needs lots of memory to build)
* Prove correctness; it may also give insights on optimal matrix shapes.
