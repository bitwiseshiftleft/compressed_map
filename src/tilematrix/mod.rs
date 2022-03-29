/*
 * @file mod.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Fast matrices using "tile" submatrices.  These only support
 * enough operations, and in a stable enough configuration, to
 * build the hierarchical matrix solver.  Possible future work:
 * Spin this off to its own mature package.
 */

/** Small fixed-size matrix "tiles" for implementing larger matrices. */
pub(crate) mod tile;

/** Fast matrix library made of tile matrices. */
pub mod matrix;

/**
 * Simple bit sets.
 * 
 * The bit-set crate is actually kind of slow,
 * and was bottlenecking the code.
 */
pub(crate) mod bitset;