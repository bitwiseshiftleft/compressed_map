# For release

* Make code more portable -- probably not full cmake though.
* At least make the test/bench frameworks consistent.
* File-handling / ser / deser API.
* Demo: CRL compression.
* FIXME BUG: nonuniform maps don't work if an item has zero population.
* Speed up builder hash table impl?
* Should we just forget murmur3/siphash and go with SHA256, hoping for HW acceleration?
* Test builder hash table impl (using C++?)
* Test the C++ interface better.
* More tests around builder expanding memory.
* Test various things with zero allowed tries, zero items, zero value bits, etc.
* Doxyfile
* Make tile_matrix_t a one-element array too?

# Longer term

* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Comprehensive unit tests.
* Make production-quality.
