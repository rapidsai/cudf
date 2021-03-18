find_path(RDKAFKA_INCLUDE "librdkafka" HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++" HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")

if(RDKAFKA_INCLUDE AND RDKAFKA++_LIBRARY)
  add_target(rdkafka INTERFACE)
  target_link_libraries(rdkafka INTERFACE "${RDKAFKA++_LIBRARY}")
  target_include_directories(rdkafka INTERFACE "${RDKAFKA_INCLUDE}")

  add_target(RDKAFKA::RDKAFKA ALIAS rdkafka)
endif()