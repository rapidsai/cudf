#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace text {

namespace {

struct trie_builder_node {
};

}  // namespace

struct trie {
  trie(std::string const& pattern) : trie(std::vector<std::string>{pattern}) {}
  trie(std::vector<std::string> const& patterns) {}
};

}  // namespace text
}  // namespace io
}  // namespace cudf
