#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <iterator>

int main() {
    std::vector<int> v = {26, 26};
    if (std::adjacent_find(v.cbegin(), v.cend(), std::not_equal_to<>()) !=
      v.cend()) {
    std::cout << "Failed" << std::endl;
  } else {
    std::cout << "Passed" << std::endl;
  }
}