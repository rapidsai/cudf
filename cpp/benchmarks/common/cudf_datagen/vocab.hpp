/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string>
#include <vector>

std::vector<std::string> const years  = {"1992", "1993", "1994", "1995", "1996", "1997", "1998"};
std::vector<std::string> const months = {
  "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"};
std::vector<std::string> const days = {
  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11", "12", "13", "14", "15", "16",
  "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"};

std::vector<std::string> const vocab_p_name = {
  "almond",   "antique",   "aquamarine", "azure",      "beige",     "bisque",    "black",
  "blanched", "blue",      "blush",      "brown",      "burlywood", "burnished", "chartreuse",
  "chiffon",  "chocolate", "coral",      "cornflower", "cornsilk",  "cream",     "cyan",
  "dark",     "deep",      "dim",        "dodger",     "drab",      "firebrick", "floral",
  "forest",   "frosted",   "gainsboro",  "ghost",      "goldenrod", "green",     "grey",
  "honeydew", "hot",       "indian",     "ivory",      "khaki",     "lace",      "lavender",
  "lawn",     "lemon",     "light",      "lime",       "linen",     "magenta",   "maroon",
  "medium",   "metallic",  "midnight",   "mint",       "misty",     "moccasin",  "navajo",
  "navy",     "olive",     "orange",     "orchid",     "pale",      "papaya",    "peach",
  "peru",     "pink",      "plum",       "powder",     "puff",      "purple",    "red",
  "rose",     "rosy",      "royal",      "saddle",     "salmon",    "sandy",     "seashell",
  "sienna",   "sky",       "slate",      "smoke",      "snow",      "spring",    "steel",
  "tan",      "thistle",   "tomato",     "turquoise",  "violet",    "wheat",     "white",
  "yellow"};

std::vector<std::string> const vocab_modes = {
  "REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"};

std::vector<std::string> const vocab_instructions = {
  "DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"};

std::vector<std::string> const vocab_priorities = {
  "1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};

std::vector<std::string> const vocab_segments = {
  "AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"};

std::vector<std::string> gen_vocab_types()
{
  std::vector<std::string> syllable_a = {
    "STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"};
  std::vector<std::string> syllable_b = {"ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"};
  std::vector<std::string> syllable_c = {"TIN", "NICKEL", "BRASS", "STEEL", "COPPER"};
  std::vector<std::string> syllable_combinations;
  for (auto const& s_a : syllable_a) {
    for (auto const& s_b : syllable_b) {
      for (auto const& s_c : syllable_c) {
        syllable_combinations.push_back(s_a + " " + s_b + " " + s_c);
      }
    }
  }
  return syllable_combinations;
}

std::vector<std::string> gen_vocab_containers()
{
  std::vector<std::string> syllable_a = {"SM", "LG", "MED", "JUMBO", "WRAP"};
  std::vector<std::string> syllable_b = {"CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"};
  std::vector<std::string> syllable_combinations;
  for (auto const& s_a : syllable_a) {
    for (auto const& s_b : syllable_b) {
      syllable_combinations.push_back(s_a + " " + s_b);
    }
  }
  return syllable_combinations;
}
