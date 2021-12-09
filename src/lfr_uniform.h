/**
 * @file lfr_uniform.h
 * @author Mike Hamburg
 * @copyright 2020-2021 Rambus Inc.
 *
 * Uniform static functions.  These objects nearly-optimally
 * encode a map of type keys -> values, where the values are
 * iid uniform values of a certain bit length.  For simplicity,
 * we support bit lengths <= 64 here; if you want to go larger
 * you probably want a perfect hashing scheme anyway.
 *
 * Building a static function may fail, with a few % probability
 * In this case it will retry up to builder->max_tries times
 * with different salts, and then fail, returning EAGAIN.  Note
 * that if you have duplicate keys, even if they have the same
 * values (e.g. by setting the builder flag LFR_NO_HASHTABLE)
 * then this will fail every time.
 * 
 * Note well! This is a research-grade library, and not ready for
 * production use.  Also, note that this library is not designed
 * to store secret data.  In particular, it doesn't employ side-
 * channel countermeasures, doesn't pin data to RAM, and doesn't
 * erase it with memset_s or the like.
 */
#ifndef __LFR_UNIFORM_H__
#define __LFR_UNIFORM_H__

#include "lfr_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

/*****************************************************************
 *                     Compiled uniform maps                     *
 *****************************************************************/

/** A compiled uniform map. */ 
typedef struct {
    size_t blocks;
    lfr_salt_t salt;
    uint8_t value_bits;
    uint8_t data_is_mine; // vector memory was allocated here, and should be deallocated with lfr_uniform_map_destroy
    uint8_t _salt_hint; // used when the salt is derived
    const uint8_t *data; // never modified but may be freed
} lfr_uniform_map_s, lfr_uniform_map_t[1];

/** High-level build function: using the builder, compile to a map object.
 *
 * @param map The map object.  On success, this function will initialize
 * the map and allocate memory for it.
 * @param builder The builder object.
 * @param value_bits The number of bits of the responses to use.  If -1, then set to max length of a required response.
 * @return 0 on success.
 * @return ENOMEM Not enough memory to solve / return the map.
 * @return EAGAIN The solution failed; either it has inconsistent values
 * or should be tried again with a different salt.
 */
int lfr_uniform_build(lfr_uniform_map_t map, const lfr_builder_t builder, int value_bits);

/** As lfr_uniform_build, but if the library was built with thread support, you
 * can set the number of threads.  Set to 0 for default.  If the library was not
 * built with thread support (by default it is not), then this call ignores
 * nthreads and always uses 1 thread.
 */
int lfr_uniform_build_threaded(lfr_uniform_map_t map, const lfr_builder_t builder, int value_bits, int nthreads);

/** Destroy a map object, and deallocate any memory used to create it. */
void lfr_uniform_map_destroy(lfr_uniform_map_t map);

/** Query a uniform map.  If the key was used when building
 * the map, then the same value will be returned.
 */
lfr_response_t lfr_uniform_query (
    const lfr_uniform_map_t map,
    const uint8_t *key,
    size_t keybytes
);

/*****************************************************************
 *                         Serialization                         *
 *****************************************************************/

/** Return the number of bytes required to serialize the map */
size_t lfr_uniform_map_serial_size(const lfr_uniform_map_t map);

/** Serialize the map.  The output should be lfr_uniform_map_ser_size(map) bytes long.
 * @return 0 on success.
 * @return nonzero on failure.  This function shouldn't fail, but maybe would
 * fail with an excessively large map.
 */
int lfr_uniform_map_serialize(uint8_t *out, const lfr_uniform_map_t map);

/**
 * Deserialize a map.  If flags & LFR_NO_COPY_DATA, then point to the data; otherwise copy it.
 * @return 0 on success.
 * @return nonzero if the map is corrupt.
 */
int lfr_uniform_map_deserialize(
    lfr_uniform_map_t map,
    const uint8_t *data,
    size_t data_size,
    uint8_t flags
);

/** Mirror of LFR_BLOCKSIZE */
extern const int _lfr_blocksize;


/*****************************************************************
 *                     Testing and debugging                     *
 *****************************************************************/

/** Return the number of bytes in the uniform map's data section. */
size_t _lfr_uniform_map_vector_size(const lfr_uniform_map_t map);

/**
 * Return the number of columns required for the given number of rows.
 * It will always be a multiple of 8*LFR_BLOCKSIZE.  Useful for sizing
 * the map.  The number of bytes required for the map's data will be
 * (columns * value_bits) / 8.
 */
size_t _lfr_uniform_provision_columns(size_t rows);

/**
 * For testing purposes.  Return the maximum number of rows such that
 * _lfr_uniform_provision_columns(rows) <= cols.  For a given number of
 * columns, the failure probability increases according to the number of
 * rows, so this is the best-case scenario in terms of compression
 * efficiency but the worst-case scenario in terms of failure probability
 * and thus speed.
 */
size_t _lfr_uniform_provision_max_rows(size_t cols);

#ifdef __cplusplus
} // extern "C"

namespace LibFrayed {
    /** Exception: couldn't build the map */
    class BuildFailedException: public std::exception {
    public:
        explicit BuildFailedException() {}
        virtual ~BuildFailedException() _NOEXCEPT {}
        virtual const char* what() const _NOEXCEPT { return ("LibFrayed map build failed"); }
    };

    /** Wrapper for map */
    class UniformMap {
    public:
        /** Wrapped map object */
        lfr_uniform_map_t map;

        /** Empty constructor */
        inline UniformMap() { memset(map,0,sizeof(map)); }

        inline UniformMap(const LibFrayed::UniformMap &other) = delete;

        /** Move constructor */
        inline UniformMap(LibFrayed::UniformMap &&other) {
            map[0] = other.map[0];
            memset(other.map,0,sizeof(other.map));
        }

        /** Deserialize from vector */
        inline UniformMap(const std::vector<uint8_t> &other, uint8_t flags=0) {
            int ret = lfr_uniform_map_deserialize(map, other.data(), other.size(), flags);
            if (ret) throw std::runtime_error("corrupt LibFrayed::UniformMap");
        }

        /** Deserialize from uint8_t* */
        inline UniformMap(const uint8_t *data, size_t data_size, uint8_t flags=0) {
            int ret = lfr_uniform_map_deserialize(map, data, data_size, flags);
            if (ret) throw std::runtime_error("corrupt LibFrayed::UniformMap");
        }

        /** Move assignment */
        inline UniformMap& operator=(LibFrayed::UniformMap &&other) {
            lfr_uniform_map_destroy(map);
            map[0] = other.map[0];
            memset(other.map,0,sizeof(other.map));
            return *this;
        }

        /** Construct from a builder */
        inline UniformMap(const LibFrayed::Builder &builder, int value_bits, int nthreads=0) {
            int ret = lfr_uniform_build_threaded(map,builder.builder,value_bits,nthreads);
            if (ret == ENOMEM) {
                throw std::bad_alloc();
            } else if (ret == EAGAIN) {
                throw BuildFailedException();
            }
        }

        /** Destructor */
        inline ~UniformMap() { lfr_uniform_map_destroy(map); }

        /** Lookup */
        inline lfr_response_t lookup(const uint8_t *data, size_t size) const {
            return lfr_uniform_query(map,data,size);
        }

        /** Lookup */
        inline lfr_response_t lookup(const std::vector<uint8_t> &v) const {
            return lookup(v.data(),v.size());
        }

        /** Lookup */
        inline lfr_response_t operator[] (const std::vector<uint8_t> &v) const {
            return lookup(v);
        }

        /** Get serial size */
        inline size_t serial_size() const { return lfr_uniform_map_serial_size(map); }
        
        /** Serialize */
        inline void serialize_into(uint8_t *out) const {
            int ret = lfr_uniform_map_serialize(out,map);
            if (ret != 0) throw std::runtime_error("LibFrayed::uniform_map::serialize_into failed");
        }

        /** Serialize and return as a vector */
        inline std::vector<uint8_t> serialize() const {
            size_t sz = serial_size();
            std::vector<uint8_t> ret(sz);
            serialize_into(ret.data());
            return ret;
        }
    };
}
#endif /* __cplusplus */

#endif // __LFR_UNIFORM_H__
