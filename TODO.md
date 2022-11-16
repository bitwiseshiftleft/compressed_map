# C FFI items

* Complete the C interface / dylib: deserialize, save/load file, map with bytes output
* Test compressed maps in CFFI
* Test (de)serialization in CFFI
* Test that CFFI doesn't leak memory
* Rename the C FFI types?

# Other v0.2 release items

* Test on Armv7 and x86, RISC-V and some big-endian system
* Demo apps
* Optimize results for small maps?  Here storing the hash key and rounding up to one block (=32 bits) are costly.
    * First possible approach: add linear constraints that certain bits of the last block must match (functions of) bits elsewhere.
        * Then encode the other bits of the last block separately in packed form, separately.
        * On decode, fill in the extra bits.
    * Second possible approach: arrange that fewer entries end up in the last block, and then directly constrain some of its bits (eg to be zero).
    * Third possible approach: make ribbon offsets in bytes (but still block-sized), and change the solver.

# Post 0.2 quality items

* Distinguish between "out of memory", "can't create thread" etc, and "matrix is not invertible"
* Enable mmap?
* Allow other implementations such as binary fuse filters?

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

* Whitepaper
* Make production-quality (1.0).
* no_std core for embedded systems
* Add SSSE3 version?  ARM SEV??
* Better interface for tile matrices; release as its own crate?
* Test on very large data sets (eg CompressedRandomMap with 1 billion entries; needs lots of memory to build)
* Prove correctness; it may also give insights on optimal matrix shapes.
