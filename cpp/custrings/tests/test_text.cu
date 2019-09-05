#include <gtest/gtest.h>
#include <vector>
#include <thrust/device_vector.h>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "./utils.h"

std::vector<const char*> tstrs{ "the fox jumped over the dog",
                                "the dog chased the cat",
                                "the cat chased the mouse",
                                nullptr, "",
                                "the mouse ate the cheese" };

TEST(TestText, Tokenize)
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

TEST(TestText, TokenCount)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    thrust::device_vector<unsigned int> results(tstrs.size(),0);

    NVText::token_count(*strs," ",results.data().get());
    unsigned int expected[] = { 6, 5, 5, 0, 0, 5 };
    for( int idx = 0; idx < (int) tstrs.size(); ++idx )
        EXPECT_EQ(results[idx],expected[idx]);

    NVStrings::destroy(strs);
}

TEST(TestText, UniqueTokens)
{
    NVStrings* strs = NVStrings::create_from_array(tstrs.data(),tstrs.size());
    NVStrings* got = NVText::unique_tokens(*strs);
    const char* expected[] = { "ate","cat","chased","cheese","dog",
                               "fox","jumped","mouse","over","the" };
    EXPECT_TRUE( verify_strings(got,expected));
    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST(TestText, ContainsStrings)
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

TEST(TestText, StringsCounts)
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

TEST(TestText, TokensCounts)
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

TEST(TestText, EditDistance)
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

TEST(TestText, NGrams)
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


TEST(TestText, PorterStemmerMeasure)
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


int main( int argc, char** argv )
{
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}