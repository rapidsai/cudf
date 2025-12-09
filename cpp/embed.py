

# create header file to embed header map
import sys
import hashlib

# [ ] include dir resolution
# [ ] get headers and options and outputs as JSON-encoded HEX strings?
# [ ] usage: python3 rtc_embed.py --embed --header --type=explicit|recursive header_map_id:reference_dir:[list of header files] --option option_id:[compiler options] --output output_file


""" headers

struct {
unsigned char const * const * include_names;
unsigned char const * const * headers;
unsigned long long const * header_sizes;
unsigned char * headers_sha256;
unsigned long long num_includes;
unsigned char sha256[32];
} ${variable_name} = { ... };

"""

"""options

struct {
unsigned char const * const * options;
unsigned char * options_sha256;
unsigned long long num_options;
unsigned char sha256[32];
} ${variable_name} = { ... };

"""

"""binary blob

struct {
unsigned char const * data;
unsigned long long size;
unsigned char sha256[32];
} ${variable_name} = { ... };

"""