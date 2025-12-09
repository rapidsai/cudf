




// SIMPLE KEY: from user
// COMPLEX KEY: sha256 of all parameters and blobs involved; + driver + runtime
//

// [ ] All these functions should have wrappers that cache results and request an optional key from
// the user

// [ ] Method to pre-compile library and reuse it across multiple operators; compile_library();
// compile_library_cached()

// [ ] jit_key(key) -> key+driver+CUDA_versions+CUDA_runtime_versions+device_compute_models
  // #include <cudf_lto_library_fatbin_bytes.h>


// [ ] environment variables to control:
// [ ] cache path
// [ ] cache entries limit
// [ ] disable caching
// [ ] cache statistics: hits, misses, etc.
// [ ] on startup, log cache path, loading information, etc.

compile_cache_t& get_cache();

// [ ] if a user provides a key, use: USER_KEY+${key}, otherwise use sha256 of contents
// [ ] use resource type in key to avoid collisions
void compile_operator();

void make_pch();

void link_operator();