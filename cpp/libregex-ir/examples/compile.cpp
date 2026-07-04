/*
 * Copyright (c) 2026, Regex IR contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <regex_ir.hpp>

#include <iostream>

int main()
{
  auto result = regex_ir::compile("abc[0-9]+", regex_ir::operation::contains());
  if (!result) {
    for (auto& diagnostic : result.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    return 1;
  }
  std::cout << regex_ir::to_string(*result.value);
}
