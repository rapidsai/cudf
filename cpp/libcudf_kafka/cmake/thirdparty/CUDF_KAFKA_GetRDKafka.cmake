find_path(RDKAFKA_INCLUDE "librdkafka" HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++" HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")