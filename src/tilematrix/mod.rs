/**
 * @file mod.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * Fast matrices using "tile" submatrices.  These only support
 * enough operations, and in a stable enough configuration, to
 * build the hierarchical matrix solver.  Possible future work:
 * Spin this off to its own mature package.
 */
pub mod tile;
pub mod matrix;
pub mod bitset;