# For release

* Make code more portable -- probably not full cmake though.
* At least make the test/bench frameworks consistent.
* File-handling / ser / deser API.
* Demo: CRL compression.
* FIXME BUG: nonuniform maps don't work if an item has zero population.
* Should uniform maps have dupe protection?
* Should uniform maps support automatic retries?
* Speed up builder hash table impl
* Test builder hash table impl (using C++?)
* Move salt setting, retries etc to builder
* Doxyfile
* C++ interface

# Longer term

* Better interface for tile matrices
* Complete multi-threading
* Test on very large data sets (eg 1 billion)
* Prove correctness
* Once correctness is proved, it may give insights on optimal matrix shapes.
* Comprehensive unit tests.
* Make production-quality.
