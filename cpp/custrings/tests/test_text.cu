#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "./utils.h"

struct TestText : public GdfTest{};

std::vector<const char*> tstrs{ "the fox jumped over the dog",
                                "the dog chased the cat",
                                "the cat chased the mouse",
                                nullptr, "",
                                "the mouse ate the cheese" };

TEST_F(TestText, Tokenize)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    NVStrings* got = NVText::tokenize(*strs);
    const char* expected[] = { "the","fox","jumped","over","the","dog",
                               "the","dog","chased","the","cat",
                               "the","cat","chased","the","mouse",
                               "the","mouse","ate","the","cheese" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestText, TokenCount)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    thrust::device_vector<unsigned int> results(tstrs.size(),0);

    NVText::token_count(*strs," ",results.data().get());
    unsigned int expected[] = { 6, 5, 5, 0, 0, 5 };
    for( int idx = 0; idx < (int) tstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestText, UniqueTokens)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    NVStrings* got = NVText::unique_tokens(*strs);
    const char* expected[] = { "ate","cat","chased","cheese","dog",
                               "fox","jumped","mouse","over","the" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestText, ContainsStrings)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    std::vector<const char*> hstrs{ "the", "cat" };
    NVStrings* tgts = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<bool> results(tstrs.size()*hstrs.size(),0);

    NVText::contains_strings(*strs,*tgts,results.data().get());
    bool expected[] = { true, false, true, true, true, true, false, false, false, false, true, false };
    for( int idx = 0; idx < (int) (tstrs.size() * hstrs.size()); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

TEST_F(TestText, StringsCounts)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    std::vector<const char*> hstrs{ "cat ", "dog " };
    NVStrings* tgts = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> results(tstrs.size()*hstrs.size(),0);

    NVText::strings_counts(*strs,*tgts,results.data().get());
    unsigned int expected[] = { 0,0, 0,1, 1,0, 0,0, 0,0, 0,0 };
    for( int idx = 0; idx < (int) (tstrs.size() * hstrs.size()); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

TEST_F(TestText, TokensCounts)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    std::vector<const char*> hstrs{ "cat", "dog" };
    NVStrings* tgts = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> results(tstrs.size()*hstrs.size(),0);

    NVText::tokens_counts(*strs,*tgts," ",results.data().get());
    unsigned int expected[] = { 0,1, 1,1, 1,0, 0,0, 0,0, 0,0 };
    for( int idx = 0; idx < (int) (tstrs.size() * hstrs.size()); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

TEST_F(TestText, CharacterTokenize)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    NVStrings* got = NVText::character_tokenize(*strs);

    const char* expected[] = { "t","h","e"," ","f","o","x"," ","j","u","m","p","e","d"," ","o","v","e","r"," ","t","h","e"," ","d","o","g",
                               "t","h","e"," ","d","o","g"," ","c","h","a","s","e","d"," ","t","h","e"," ","c","a","t",
                               "t","h","e"," ","c","a","t"," ","c","h","a","s","e","d"," ","t","h","e"," ","m","o","u","s","e",
                               "t","h","e"," ","m","o","u","s","e"," ","a","t","e"," ","t","h","e"," ","c","h","e","e","s","e" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestText, EditDistance)
{
    std::vector<const char*> hstrs{ "dog", nullptr, "cat", "mouse",
                                    "pup", "", "puppy" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());
    thrust::device_vector<unsigned int> results(hstrs.size(),0);
    {
        NVText::edit_distance(NVText::levenshtein,*strs,"puppy",results.data().get());
        unsigned int expected[] = { 5,5,5,5,2,5,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    {
        std::vector<const char*> htgts{ "hog", "not", "cake", "house",
                                        "fox", nullptr, "puppy" };
        NVStrings* tgts = NVStrings::create_from_array(htgts.data(),htgts.size());
        NVText::edit_distance(NVText::levenshtein,*strs,*tgts,results.data().get());
        unsigned int expected[] = { 1,3,2,1,3,0,0 };
        for( int idx = 0; idx < (int) hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
        NVStrings::destroy(tgts);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestText, EditDistanceMatrix)
{
    std::vector<const char*> hstrs{ "dog", nullptr, "cat", "mouse",
                                    "pup", "", "puppy" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> results(hstrs.size()*hstrs.size(),0);

    NVText::edit_distance_matrix(NVText::levenshtein,*strs,results.data().get());
    unsigned int expected[] = { 0,3,3,4,3,3,5,
                                3,0,3,5,3,0,5,
                                3,3,0,5,3,3,5,
                                4,5,5,0,4,5,5,
                                3,3,3,4,0,3,2,
                                3,0,3,5,3,0,5,
                                5,5,5,5,2,5,0};
    for( int idx = 0; idx < (int) (hstrs.size()*hstrs.size()); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);
    NVStrings::destroy(strs);
}

TEST_F(TestText, NGrams)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    NVStrings* tkns = NVText::tokenize(*strs);
    NVStrings* got = NVText::create_ngrams(*tkns,2,"_");

    const char* expected[] = { "the_fox","fox_jumped","jumped_over",
                               "over_the","the_dog","dog_the","the_dog",
                               "dog_chased", "chased_the", "the_cat",
                               "cat_the", "the_cat", "cat_chased",
                               "chased_the", "the_mouse", "mouse_the",
                               "the_mouse", "mouse_ate", "ate_the",
                               "the_cheese" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(tkns);
    NVStrings::destroy(strs);
}

TEST_F(TestText, NGramsTokenize)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    {
        NVStrings* got = NVText::ngrams_tokenize(*strs," ",2,"_");
        const char* expected[] = { "the_fox", "fox_jumped", "jumped_over", "over_the", "the_dog",
                                   "the_dog", "dog_chased", "chased_the", "the_cat",
                                   "the_cat", "cat_chased", "chased_the", "the_mouse",
                                   "the_mouse", "mouse_ate", "ate_the", "the_cheese" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    {
        NVStrings* got = NVText::ngrams_tokenize(*strs,nullptr,3,"_");
        const char* expected[] = { "the_fox_jumped", "fox_jumped_over", "jumped_over_the", "over_the_dog",
                                   "the_dog_chased", "dog_chased_the",  "chased_the_cat",
                                   "the_cat_chased", "cat_chased_the",  "chased_the_mouse",
                                   "the_mouse_ate",  "mouse_ate_the",   "ate_the_cheese" };
        EXPECT_TRUE( verify_strings(got,expected));
        NVStrings::destroy(got);
    }
    NVStrings::destroy(strs);
}

TEST_F(TestText, PorterStemmerMeasure)
{
    std::vector<const char*> hstrs{ "abandon", nullptr, "abbey", "cleans",
                                    "trouble", "", "yearly" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<unsigned int> results(hstrs.size(),0);
    NVText::porter_stemmer_measure(*strs, nullptr, nullptr, results.data().get());
    unsigned int expected[] = { 3, 0, 2, 1, 1, 0, 1 };
    for( int idx = 0; idx < (int) hstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST_F(TestText, VowelsAndConsonants)
{
    std::vector<const char*> hstrs{ "abandon", nullptr, "abbey", "cleans",
                                    "trouble", "", "yearly" };
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    thrust::device_vector<bool> results(hstrs.size(),0);
    {
        NVText::is_letter(*strs, nullptr, nullptr, NVText::vowel, 5, results.data().get());
        bool expected[] = { true, false, false, false, false, false, true };
        for( unsigned int idx=0; idx < hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    {
        NVText::is_letter(*strs, nullptr, nullptr, NVText::consonant, 5, results.data().get());
        bool expected[] = { false, false, false, true, true, false, false };
        for( unsigned int idx=0; idx < hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    thrust::device_vector<int> indices(hstrs.size());
    thrust::sequence( thrust::device, indices.begin(), indices.end() );
    indices[hstrs.size()-1] = -1; // throw in a negative index too
    {
        NVText::is_letter(*strs, nullptr, nullptr, NVText::vowel, indices.data().get(), results.data().get());
        bool expected[] = { true, false, false, true, false, false, true };
        for( unsigned int idx=0; idx < hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }
    {
        NVText::is_letter(*strs, nullptr, nullptr, NVText::consonant, indices.data().get(), results.data().get());
        bool expected[] = { false, false, true, false, true, false, false };
        for( unsigned int idx=0; idx < hstrs.size(); ++idx )
            EXPECT_EQ(results[idx],expected[idx]);
    }

    NVStrings::destroy(strs);
}

TEST_F(TestText, ScatterCount)
{
    std::vector<const char*> names{ "Larry", "Curly", "Moe" };
    NVStrings* strs = NVStrings::create_from_array(names.data(),names.size());
    unsigned int hcounts[] = { 3,0,1 };
    thrust::device_vector<unsigned int> counts(strs->size(),0);
    for( unsigned int idx=0; idx < strs->size(); ++idx )
        counts[idx] = hcounts[idx];
    NVStrings* got = NVText::scatter_count(*strs,counts.data().get());

    const char* expected[] = { "Larry", "Larry", "Larry", "Moe" };
    EXPECT_TRUE( verify_strings(got,expected));

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}
