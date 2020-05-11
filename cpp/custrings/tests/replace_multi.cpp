#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdio>
#include <random>
#include <vector>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

//
// cd ../build
// g++ -std=c++11 ../tests/replace_multi.cpp -I/usr/local/cuda/include -L. -lNVStrings -lNVText -o
// replace_multi -Wl,-rpath,.:
//

double GetTime()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.0;
}

void test1()
{
  const char* hstrs[] = {"hello there, good friend!", "hi there!", nullptr, "", "!accénted"};
  int count           = 5;
  NVStrings* strs     = NVStrings::create_from_array(hstrs, count);
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print();

  const char* htgts[] = {",", "!", "e"};
  int tcount          = 3;
  NVStrings* tgts     = NVStrings::create_from_array(htgts, tcount);
  printf("targets(%d): (%ld bytes)\n", tgts->size(), tgts->memsize());
  tgts->print();

  const char* hrpls[] = {"_"};
  unsigned int rcount = 1;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  NVStrings* result = strs->replace(*tgts, *rpls);
  printf("result: (%ld bytes)\n", result->memsize());
  result->print();

  const char* hresult[] = {"h_llo th_r__ good fri_nd_", "hi th_r__", nullptr, "", "_accént_d"};
  // verify result against hresult
  // can use NVStrings:match_strings to compare two instances
  // add up boolean values to check against count

  NVStrings::destroy(result);
  NVStrings::destroy(tgts);
  NVStrings::destroy(strs);
}

void test2()
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(32, 126);
  std::vector<std::string> data;
  std::vector<const char*> data_ptrs;
  for (int idx = 0; idx < 1000000; ++idx) {
    std::string str;
    for (int jdx = 0; jdx < 20; ++jdx) {
      char ch = (char)dist(mt);
      str.append(1, ch);
    }
    data.push_back(str);
    data_ptrs.push_back(data[data.size() - 1].c_str());
  }

  NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);
  NVStrings* tgts = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size() / 10);
  printf("targets(%d): (%ld bytes)\n", tgts->size(), tgts->memsize());
  tgts->print(0, 10);
  const char* hrpls[] = {"_"};
  unsigned int rcount = 1;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  {
    double st         = GetTime();
    NVStrings* result = strs->replace(*tgts, *rpls);
    double et         = GetTime() - st;
    printf("result: (%ld bytes)\n", result->memsize());
    result->print(0, 10);
    printf("  %g seconds\n", et);
    NVStrings::destroy(result);
  }

  // verify that the first size()/10 strings are all just "_"
  // - can use NVStrings:match_strings to compare two instances
  //   add up boolean values to check against count
  // - need to build NVStrings instance with "_" strings in it

  NVStrings::destroy(tgts);
  NVStrings::destroy(strs);
}

void test_regex1()
{
  const char* hstrs[] = {"hello @abc @def world",
                         "the quick brown @fox jumps over the",
                         "lazy @dog",
                         "hello http://www.world.com were there @home"};
  int count           = 4;
  NVStrings* strs     = NVStrings::create_from_array(hstrs, count);
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print();
  const char* hptns[] = {"@\\S+", "\\bthe\\b"};
  int tcount          = 2;
  printf("patterns(%d): (cpu)\n", tcount);
  std::vector<const char*> ptns;
  for (int i = 0; i < tcount; ++i) {
    printf("%d:[%s]\n", i, hptns[i]);
    ptns.push_back(hptns[i]);
  }

  const char* hrpls[] = {"***", ""};
  int rcount          = 1;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  NVStrings* result = strs->replace_re(ptns, *rpls);
  printf("result: (%ld bytes)\n", result->memsize());
  result->print();

  // verify result against hresult
  // can use NVStrings:match_strings to compare two instances
  // add up boolean values to check against count

  NVStrings::destroy(result);
  NVStrings::destroy(rpls);
  NVStrings::destroy(strs);
}

void test_regex2()
{
  const char* hstrs[] = {"the quick brown fox jumps over the lazy dog",
                         "the fat cat lays next to the other accénted cat",
                         "a slow moving turtlé cannot catch the bird",
                         "which can be composéd together to form a more complete",
                         "thé result does not include the value in the sum in",
                         "",
                         "absent stop words"};

  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_int_distribution<int> dist(0,7);
  std::vector<const char*> data_ptrs;
  for (int idx = 0; idx < 1000000; ++idx) data_ptrs.push_back(hstrs[idx % 7]);  // dist(mt)
  NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);

  const char* stop_words_re[] = {
    "\\bi\\b",       "\\bme\\b",       "\\bmy\\b",         "\\bmyself\\b",     "\\bwe\\b",
    "\\bour\\b",     "\\bours\\b",     "\\bourselves\\b",  "\\byou\\b",        "\\byour\\b",
    "\\byours\\b",   "\\byourself\\b", "\\byourselves\\b", "\\bhe\\b",         "\\bhim\\b",
    "\\bhis\\b",     "\\bhimself\\b",  "\\bshe\\b",        "\\bher\\b",        "\\bhers\\b",
    "\\bherself\\b", "\\bit\\b",       "\\bits\\b",        "\\bitself\\b",     "\\bthey\\b",
    "\\bthem\\b",    "\\btheir\\b",    "\\btheirs\\b",     "\\bthemselves\\b", "\\bwhat\\b",
    "\\bwhich\\b",   "\\bwho\\b",      "\\bwhom\\b",       "\\bthis\\b",       "\\bthat\\b",
    "\\bthese\\b",   "\\bthose\\b",    "\\bam\\b",         "\\bis\\b",         "\\bare\\b",
    "\\bwas\\b",     "\\bwere\\b",     "\\bbe\\b",         "\\bbeen\\b",       "\\bbeing\\b",
    "\\bhave\\b",    "\\bhas\\b",      "\\bhad\\b",        "\\bhaving\\b",     "\\bdo\\b",
    "\\bdoes\\b",    "\\bdid\\b",      "\\bdoing\\b",      "\\ba\\b",          "\\ban\\b",
    "\\bthe\\b",     "\\band\\b",      "\\bbut\\b",        "\\bif\\b",         "\\bor\\b",
    "\\bbecause\\b", "\\bas\\b",       "\\buntil\\b",      "\\bwhile\\b",      "\\bof\\b",
    "\\bat\\b",      "\\bby\\b",       "\\bfor\\b",        "\\bwith\\b",       "\\babout\\b",
    "\\bagainst\\b", "\\bbetween\\b",  "\\binto\\b",       "\\bthrough\\b",    "\\bduring\\b",
    "\\bbefore\\b",  "\\bafter\\b",    "\\babove\\b",      "\\bbelow\\b",      "\\bto\\b",
    "\\bfrom\\b",    "\\bup\\b",       "\\bdown\\b",       "\\bin\\b",         "\\bout\\b",
    "\\bon\\b",      "\\boff\\b",      "\\bover\\b",       "\\bunder\\b",      "\\bagain\\b",
    "\\bfurther\\b", "\\bthen\\b",     "\\bonce\\b",       "\\bhere\\b",       "\\bthere\\b",
    "\\bwhen\\b",    "\\bwhere\\b",    "\\bwhy\\b",        "\\bhow\\b",        "\\ball\\b",
    "\\bany\\b",     "\\bboth\\b",     "\\beach\\b",       "\\bfew\\b",        "\\bmore\\b",
    "\\bmost\\b",    "\\bother\\b",    "\\bsome\\b",       "\\bsuch\\b",       "\\bno\\b",
    "\\bnor\\b",     "\\bnot\\b",      "\\bonly\\b",       "\\bown\\b",        "\\bsame\\b",
    "\\bso\\b",      "\\bthan\\b",     "\\btoo\\b",        "\\bvery\\b",       "\\bs\\b",
    "\\bt\\b",       "\\bcan\\b",      "\\bwill\\b",       "\\bjust\\b",       "\\bdon\\b",
    "\\bshould\\b",  "\\bnow\\b",      "\\buses\\b",       "\\buse\\b",        "\\busing\\b",
    "\\bused\\b",    "\\bone\\b",      "\\balso\\b"};
  unsigned int tcount = 133;
  printf("patterns(%d): (cpu)\n", tcount);
  std::vector<const char*> ptns;
  for (int idx = 0; idx < tcount; ++idx) {
    if (idx < 10) printf("%d:[%s]\n", idx, stop_words_re[idx]);
    ptns.push_back(stop_words_re[idx]);
  }

  const char* hrpls[] = {""};
  unsigned int rcount = 1;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  double st         = GetTime();
  NVStrings* result = strs->replace_re(ptns, *rpls);
  double et         = GetTime() - st;
  printf("result: (%ld bytes)\n", result->memsize());
  result->print(0, 10);
  printf("  %g seconds\n", et);
  NVStrings::destroy(result);

  NVStrings::destroy(rpls);
  NVStrings::destroy(strs);
}

void test_text1()
{
  const char* hstrs[] = {"the quick brown fox jumps over the lazy dog",
                         "the fat cat lays next to the other accénted cat",
                         "a slow moving turtlé cannot catch the bird",
                         "which can be composéd together to form a more complete",
                         "thé result does not include the value in the sum in",
                         "",
                         "absent stop words"};
  size_t count        = 7;
  NVStrings* strs     = NVStrings::create_from_array(hstrs, count);
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);

  const char* htgts[] = {"the", "a", "in", "to", "be", "not"};
  unsigned int tcount = 6;
  NVStrings* tgts     = NVStrings::create_from_array(htgts, tcount);
  printf("tokens(%d): (%ld bytes)\n", tgts->size(), tgts->memsize());
  tgts->print();

  const char* hrpls[] = {"1", "2", "3", "4", "5", "6"};
  unsigned int rcount = 6;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  NVStrings* result = NVText::replace_tokens(*strs, *tgts, *rpls);
  printf("result: (%ld bytes)\n", result->memsize());
  result->print(0, 10);
  NVStrings::destroy(result);
  NVStrings::destroy(rpls);
  NVStrings::destroy(tgts);
  NVStrings::destroy(strs);
}

void test_text2()
{
  const char* hstrs[] = {"the quick brown fox jumps over the lazy dog",
                         "the fat cat lays next to the other accénted cat",
                         "a slow moving turtlé cannot catch the bird",
                         "which can be composéd together to form a more complete",
                         "thé result does not include the value in the sum in",
                         "",
                         "absent stop words"};

  std::vector<const char*> data_ptrs;
  for (int idx = 0; idx < 1000000; ++idx) data_ptrs.push_back(hstrs[idx % 7]);
  NVStrings* strs = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("strings(%d): (%ld bytes)\n", strs->size(), strs->memsize());
  strs->print(0, 10);

  const char* stop_words[] = {
    "i",       "me",      "my",      "myself",   "we",         "our",    "ours",    "ourselves",
    "you",     "your",    "yours",   "yourself", "yourselves", "he",     "him",     "his",
    "himself", "she",     "her",     "hers",     "herself",    "it",     "its",     "itself",
    "they",    "them",    "their",   "theirs",   "themselves", "what",   "which",   "who",
    "whom",    "this",    "that",    "these",    "those",      "am",     "is",      "are",
    "was",     "were",    "be",      "been",     "being",      "have",   "has",     "had",
    "having",  "do",      "does",    "did",      "doing",      "a",      "an",      "the",
    "and",     "but",     "if",      "or",       "because",    "as",     "until",   "while",
    "of",      "at",      "by",      "for",      "with",       "about",  "against", "between",
    "into",    "through", "during",  "before",   "after",      "above",  "below",   "to",
    "from",    "up",      "down",    "in",       "out",        "on",     "off",     "over",
    "under",   "again",   "further", "then",     "once",       "here",   "there",   "when",
    "where",   "why",     "how",     "all",      "any",        "both",   "each",    "few",
    "more",    "most",    "other",   "some",     "such",       "no",     "nor",     "not",
    "only",    "own",     "same",    "so",       "than",       "too",    "very",    "s",
    "t",       "can",     "will",    "just",     "don",        "should", "now",     "uses",
    "use",     "using",   "used",    "one",      "also"};

  unsigned int tcount = 133;
  data_ptrs.clear();
  ;
  for (int idx = 0; idx < tcount; ++idx) data_ptrs.push_back(stop_words[idx]);
  NVStrings* tgts = NVStrings::create_from_array(data_ptrs.data(), data_ptrs.size());
  printf("patterns(%d): (%ld bytes)\n", tgts->size(), tgts->memsize());
  tgts->print(0, 10);

  const char* hrpls[] = {""};
  unsigned int rcount = 1;
  NVStrings* rpls     = NVStrings::create_from_array(hrpls, rcount);
  printf("repls(%d): (%ld bytes)\n", rpls->size(), rpls->memsize());
  rpls->print();

  double st = GetTime();
  NVStrings* result =
    NVText::replace_tokens(*strs, *tgts, *rpls, "\t\f\n ");  // delimiter default is whitespace
  double et = GetTime() - st;
  printf("result: (%ld bytes)\n", result->memsize());
  result->print(0, 10);
  printf("  %g seconds\n", et);
  NVStrings::destroy(result);

  NVStrings::destroy(rpls);
  NVStrings::destroy(tgts);
  NVStrings::destroy(strs);
}

int main(int argc, char** argv)
{
  printf("---modify---------------------------------------\n");
  test1();
  test2();
  printf("---regex----------------------------------------\n");
  test_regex1();
  test_regex2();
  printf("---token----------------------------------------\n");
  test_text1();
  test_text2();

  return 0;
}
