# Release items

* Complete the C interface / dylib: deserialize, save/load file, map with bytes output
* Test CFFI
* Test on Armv7 and x86
* Demo apps

# Post 0.2 quality items

* Distinguish between "out of memory", "can't create thread" etc, and "matrix is not invertible"
* Deal with overflow cases with billions of items in nonuniform maps, where 0 rounds up to 1.
* Enable mmap?

# Performance

* Reduce C dylib size?
* Make a one-shot C vector to Rust map compression call.
* Why is SipHasher so slow on Intel?
* Multithread hashing even if we aren't multithreading bucketsort.  (Using Rayon??)
* Improve optimization of the threaded version
* Improve pseudoinverse
* More profile-driven optimization
* Compare which parts are still faster in C
* Somehow LTO speeds up nonuniform maps but regresses uniform query.

# Longer term

* Make production-quality (1.0).
* no_std core for embedded systems
* Add SSSE3 version?  ARM SEV??
* Better interface for tile matrices; release as its own crate?
* Test on very large data sets (eg CompressedRandomMap with 1 billion entries; needs lots of memory to build)
* Prove correctness; it may also give insights on optimal matrix shapes.
