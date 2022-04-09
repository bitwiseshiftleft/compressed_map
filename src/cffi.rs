/*
 * @file cffi.rs
 * @author Mike Hamburg
 * @copyright 2020-2022 Rambus Inc.
 *
 * C foreign function interface.
 * All boilerplate. Would be nice if GPT3 / macros could write this :-/
 */

use crate::{BuildOptions,CompressedRandomMap,ApproxSet,STD_BINCODE_CONFIG,CompressedMap,serialized_size};
use std::collections::{HashSet,HashMap};
use core::ptr::NonNull;
use core::slice::from_raw_parts_mut;
use bincode::encode_into_slice;

/// Rust version of a vector of bytes
pub type Bytes = Box<[u8]>;

unsafe fn ptr_to_bytes(ptr: *const u8, len: usize) -> Bytes {
    std::slice::from_raw_parts(ptr,len).to_vec().into_boxed_slice()
}

/****************************************************************************
 * bytes -> u64
 ****************************************************************************/

#[no_mangle]
/// Create new HashMap
pub extern fn cmap_hashmap_bytes_u64_new() -> *mut HashMap<Bytes,u64> {
    Box::into_raw(Box::new(HashMap::new()))
}

#[no_mangle]
/// Count the items in a HashMap
pub unsafe extern fn cmap_hashmap_bytes_u64_len(ptr: NonNull<HashMap<Bytes,u64>>) -> usize {
    ptr.as_ref().len()
}

#[no_mangle]
/// Insert an item into a HashMap
pub unsafe extern fn cmap_hashmap_bytes_u64_insert(mut ptr: NonNull<HashMap<Bytes,u64>>,
        key: *const u8, key_len: usize, value:u64) {
    ptr.as_mut().insert(ptr_to_bytes(key,key_len),value);
}

#[no_mangle]
/// Remove an item from a HashMap
pub unsafe extern fn cmap_hashmap_bytes_u64_remove(mut ptr: NonNull<HashMap<Bytes,u64>>,
        key: *const u8, key_len: usize) {
    ptr.as_mut().remove(&ptr_to_bytes(key,key_len));
}

#[no_mangle]
/// Does this HashMap contain a given key?
pub unsafe extern fn cmap_hashmap_bytes_u64_contains(mut ptr: NonNull<HashMap<Bytes,u64>>,
        key: *const u8, key_len: usize) -> bool {
    ptr.as_mut().contains_key(&ptr_to_bytes(key,key_len))
}

#[no_mangle]
/// Look up a key in a hashmap.  Return true if it contains the key
pub unsafe extern fn cmap_hashmap_bytes_u64_get(ptr: NonNull<HashMap<Bytes,u64>>,
        key: *const u8, key_len: usize, output: *mut u64) -> bool {
    ptr.as_ref().get(&ptr_to_bytes(key,key_len)).map_or(false, |v| {
        if !output.is_null() { *output = *v; }
        true
    })
}

#[no_mangle]
/// Free a HashMap
pub unsafe extern fn cmap_hashmap_bytes_u64_free(ptr: *mut HashMap<Bytes,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build a CompressedMap.  
pub unsafe extern fn cmap_compressed_map_bytes_u64_build<'a>(ptr: NonNull<HashMap<Bytes,u64>>)
        -> *mut CompressedMap<'a, Bytes, u64> {
    let mut options = BuildOptions::default();
    if let Some(cmap) = CompressedMap::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(cmap));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Look up a key in a CompressedMap
pub unsafe extern fn cmap_compressed_map_bytes_u64_query<'a>(
    ptr: NonNull<CompressedMap<'a, Bytes,u64>>,
    key: *const u8, key_len: usize
) -> u64 {
    ptr.as_ref()[&ptr_to_bytes(key,key_len)]
}

#[no_mangle]
/// Encode to output_buf, if it's big enough.  Return the serialized size of the object, in bytes.
pub unsafe extern fn cmap_compressed_map_bytes_u64_encode<'a>(
    ptr: NonNull<CompressedMap<'a,Bytes,u64>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Destroy and free a CompressedMap
pub unsafe extern fn cmap_compressed_map_bytes_u64_free<'a>(ptr: *mut CompressedMap<'a,Bytes,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build a CompressedRandomMap.  Return NULL on failure
pub unsafe extern fn cmap_compressed_random_map_bytes_u64_build<'a>(ptr: NonNull<HashMap<Bytes,u64>>)
        -> *mut CompressedRandomMap<'a, Bytes, u64> {
    let mut options = BuildOptions::default();
    if let Some(cmap) = CompressedRandomMap::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(cmap));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Look up a key in a CompressedRandomMap
pub unsafe extern fn cmap_compressed_random_map_bytes_u64_query<'a>(
    ptr: NonNull<CompressedRandomMap<'a, Bytes,u64>>,
    key: *const u8, key_len: usize
) -> u64 {
    ptr.as_ref().query(&ptr_to_bytes(key,key_len))
}

#[no_mangle]
/// Encode to output_buf, if it's big enough.  Return the serialized size of the object, in bytes.
pub unsafe extern fn cmap_compressed_random_map_bytes_u64_encode<'a>(
    ptr: NonNull<CompressedRandomMap<'a,Bytes,u64>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Destroy and free a CompressedRandomMap
pub unsafe extern fn cmap_compressed_random_map_bytes_u64_free<'a>(ptr: *mut CompressedRandomMap<'a,Bytes,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}


/****************************************************************************
 * u64 -> u64
 ****************************************************************************/


#[no_mangle]
/// Create new HashMap
pub extern fn cmap_hashmap_u64_u64_new() -> *mut HashMap<u64,u64> {
    Box::into_raw(Box::new(HashMap::new()))
}

#[no_mangle]
/// Count the items in a HashMap
pub unsafe extern fn cmap_hashmap_u64_u64_len(ptr: NonNull<HashMap<u64,u64>>) -> usize {
    ptr.as_ref().len()
}

#[no_mangle]
/// Insert an item into a HashMap
pub unsafe extern fn cmap_hashmap_u64_u64_insert(mut ptr: NonNull<HashMap<u64,u64>>,
        key: u64, value:u64) {
    ptr.as_mut().insert(key,value);
}

#[no_mangle]
/// Remove an item from a HashMap
pub unsafe extern fn cmap_hashmap_u64_u64_remove(mut ptr: NonNull<HashMap<u64,u64>>,
        key: u64) {
    ptr.as_mut().remove(&key);
}

#[no_mangle]
/// Does this HashMap contain a given key?
pub unsafe extern fn cmap_hashmap_bu64_u64_contains(mut ptr: NonNull<HashMap<u64,u64>>,
        key: u64) -> bool {
    ptr.as_mut().contains_key(&key)
}


#[no_mangle]
/// Look up a key in a hashmap.  Return true if it contains the key
pub unsafe extern fn cmap_hashmap_u64_u64_get(ptr: NonNull<HashMap<u64,u64>>,
        key: u64, output: *mut u64) -> bool {
    ptr.as_ref().get(&key).map_or(false, |v| {
        if !output.is_null() { *output = *v; }
        true
    })
}

#[no_mangle]
/// Free a HashMap
pub unsafe extern fn cmap_hashmap_u64_u64_free(ptr: *mut HashMap<u64,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build a CompressedMap.  
pub unsafe extern fn cmap_compressed_map_u64_u64_build<'a>(ptr: NonNull<HashMap<u64,u64>>)
        -> *mut CompressedMap<'a, u64, u64> {
    let mut options = BuildOptions::default();
    if let Some(cmap) = CompressedMap::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(cmap));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Look up a key in a CompressedMap
pub unsafe extern fn cmap_compressed_map_u64_u64_query<'a>(
    ptr: NonNull<CompressedMap<'a, u64,u64>>,
    key: u64
) -> u64 {
    ptr.as_ref()[&key]
}

#[no_mangle]
/// Encode to output_buf, if it's big enough.  Return the serialized size of the object, in bytes.
pub unsafe extern fn cmap_compressed_map_u64_u64_encode<'a>(
    ptr: NonNull<CompressedMap<'a,u64,u64>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Destroy and free a CompressedMap
pub unsafe extern fn cmap_compressed_map_u64_u64_free<'a>(ptr: *mut CompressedMap<'a,u64,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build a CompressedRandomMap.  Return NULL on failure
pub unsafe extern fn cmap_compressed_random_map_u64_u64_build<'a>(ptr: NonNull<HashMap<u64,u64>>)
        -> *mut CompressedRandomMap<'a, u64, u64> {
    let mut options = BuildOptions::default();
    if let Some(cmap) = CompressedRandomMap::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(cmap));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Return serialized size of the map, in bytes
pub unsafe extern fn cmap_compressed_random_map_u64_u64_encode<'a>(
    ptr: NonNull<CompressedRandomMap<u64,u64>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Look up a key in a CompressedRandomMap
pub unsafe extern fn cmap_compressed_random_map_u64_u64_query<'a>(
    ptr: NonNull<CompressedRandomMap<'a, u64,u64>>,
    key: u64
) -> u64 {
    ptr.as_ref().query(&key)
}

#[no_mangle]
/// Destroy and free a CompressedRandomMap
pub unsafe extern fn cmap_compressed_random_map_u64_u64_free<'a>(ptr: *mut CompressedRandomMap<'a,u64,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}


/****************************************************************************
 * HashSet<Bytes>
 ****************************************************************************/

#[no_mangle]
/// Create new HashSet
pub extern fn cmap_hashset_bytes_new() -> *mut HashSet<Bytes> {
    Box::into_raw(Box::new(HashSet::new()))
}

#[no_mangle]
/// Count the items in a HashMap
pub unsafe extern fn cmap_hashset_bytes_len(ptr: NonNull<HashSet<Bytes>>) -> usize {
    ptr.as_ref().len()
}

#[no_mangle]
/// Insert an item into a HashSet
pub unsafe extern fn cmap_hashset_bytes_insert(mut ptr: NonNull<HashSet<Bytes>>,
        key: *const u8, key_len: usize) {
    ptr.as_mut().insert(ptr_to_bytes(key,key_len));
}

#[no_mangle]
/// Remove an item from a HashSet
pub unsafe extern fn cmap_hashset_bytes_remove(mut ptr: NonNull<HashSet<Bytes>>,
        key: *const u8, key_len: usize) {
    ptr.as_mut().remove(&ptr_to_bytes(key,key_len));
}

#[no_mangle]
/// Does this HashSet contain a key
pub unsafe extern fn cmap_hashset_bytes_contains(mut ptr: NonNull<HashSet<Bytes>>,
        key: *const u8, key_len: usize) -> bool {
    ptr.as_mut().contains(&ptr_to_bytes(key,key_len))
}

#[no_mangle]
/// Free a HashSet
pub unsafe extern fn cmap_hashset_bytes_free(ptr: *mut HashSet<Bytes>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build an ApproxSet.  Return NULL on failure
pub unsafe extern fn cmap_approxset_bytes_build<'a>(ptr: NonNull<HashSet<Bytes>>)
        -> *mut ApproxSet<'a, Bytes> {
    let mut options = BuildOptions::default();
    if let Some(aset) = ApproxSet::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(aset));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Look up a key in an ApproxSet
pub unsafe extern fn cmap_approxset_bytes_probably_contains<'a>(
    ptr: NonNull<ApproxSet<'a, Bytes>>,
    key: *const u8, key_len: usize
) -> bool {
    ptr.as_ref().probably_contains(&ptr_to_bytes(key,key_len))
}

#[no_mangle]
/// Encode to output_buf, if it's big enough.  Return the serialized size of the object, in bytes.
pub unsafe extern fn cmap_approxset_bytes_encode<'a>(
    ptr: NonNull<ApproxSet<'a,Bytes>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Destroy and free an ApproxSet
pub unsafe extern fn cmap_approxset_bytes_free<'a>(ptr: *mut ApproxSet<'a,Bytes>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

/****************************************************************************
 * HashSet<u64>
 ****************************************************************************/

#[no_mangle]
/// Create new HashSet
pub extern fn cmap_hashset_u64_new() -> *mut HashSet<u64> {
    Box::into_raw(Box::new(HashSet::new()))
}

#[no_mangle]
/// Count the items in a HashMap
pub unsafe extern fn cmap_hashset_u64_len(ptr: NonNull<HashSet<u64>>) -> usize {
    ptr.as_ref().len()
}

#[no_mangle]
/// Insert an item into a HashSet
pub unsafe extern fn cmap_hashset_u64_insert(mut ptr: NonNull<HashSet<u64>>, key: u64) {
    ptr.as_mut().insert(key);
}

#[no_mangle]
/// Remove an item from a HashSet
pub unsafe extern fn cmap_hashset_u64_remove(mut ptr: NonNull<HashSet<u64>>, key: u64) {
    ptr.as_mut().remove(&key);
}

#[no_mangle]
/// Does this HashSet contain a key
pub unsafe extern fn cmap_hashset_u64_contains(mut ptr: NonNull<HashSet<u64>>, key: u64) -> bool {
    ptr.as_mut().contains(&key)
}

#[no_mangle]
/// Free a HashSet
pub unsafe extern fn cmap_hashset_u64_free(ptr: *mut HashSet<u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}

#[no_mangle]
/// Build an ApproxSet.  Return NULL on failure
pub unsafe extern fn cmap_approxset_u64_build<'a>(ptr: NonNull<HashSet<u64>>)
        -> *mut ApproxSet<'a, u64> {
    let mut options = BuildOptions::default();
    if let Some(aset) = ApproxSet::build(ptr.as_ref(),&mut options) {
        return Box::into_raw(Box::new(aset));
    }
    std::ptr::null_mut()
}

#[no_mangle]
/// Look up a key in an ApproxSet
pub unsafe extern fn cmap_approxset_u64_probably_contains<'a>(
    ptr: NonNull<ApproxSet<'a, u64>>, key:u64
) -> bool {
    ptr.as_ref().probably_contains(&key)
}

#[no_mangle]
/// Encode to output_buf, if it's big enough.  Return the serialized size of the object, in u64.
pub unsafe extern fn cmap_approxset_u64_encode<'a>(
    ptr: NonNull<ApproxSet<'a,u64>>,
    output_buf: *mut u8,
    output_buf_size: usize
) -> usize {
    let required_size = serialized_size(ptr.as_ref(),STD_BINCODE_CONFIG).unwrap();
    if required_size <= output_buf_size && !output_buf.is_null() {
        encode_into_slice(ptr.as_ref(), from_raw_parts_mut(output_buf,output_buf_size), STD_BINCODE_CONFIG).unwrap();
    }
    required_size
}

#[no_mangle]
/// Destroy and free an ApproxSet
pub unsafe extern fn cmap_approxset_u64_free<'a>(ptr: *mut ApproxSet<'a,u64>) {
    if !ptr.is_null() { Box::from_raw(ptr); }
}
 