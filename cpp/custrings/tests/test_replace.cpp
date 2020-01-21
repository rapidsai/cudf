#include <gtest/gtest.h>
#include <vector>

#include "nvstrings/NVStrings.h"
#include "nvstrings/NVText.h"

#include "./utils.h"

struct TestReplace : public GdfTest{};

std::vector<const char*> hstrs{ "the quick brown fox jumps over the lazy dog",
                                "the fat cat lays next to the other accénted cat",
                                 "a slow moving turtlé cannot catch the bird",
                                 "which can be composéd together to form a more complete",
                                 "thé result does not include the value in the sum in",
                                 "", "absent stop words" };

TEST_F(TestReplace, Replace)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());


    NVStrings* got = strs->replace("the ", "++++ ");

    std::vector<const char*> expected{ "++++ quick brown fox jumps over ++++ lazy dog",
                                "++++ fat cat lays next to ++++ other accénted cat",
                                 "a slow moving turtlé cannot catch ++++ bird",
                                 "which can be composéd together to form a more complete",
                                 "thé result does not include ++++ value in ++++ sum in",
                                 "", "absent stop words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestReplace, ReplaceRegex)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());


    NVStrings* got = strs->replace_re("(\\bin\\b)|(\\ba\\b)|(\\bthe\\b)", "=");

    std::vector<const char*> expected{ "= quick brown fox jumps over = lazy dog",
                                 "= fat cat lays next to = other accénted cat",
                                 "= slow moving turtlé cannot catch = bird",
                                 "which can be composéd together to form = more complete",
                                 "thé result does not include = value = = sum =",
                                 "", "absent stop words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}

TEST_F(TestReplace, ReplaceMulti)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    std::vector<const char*> htgts{ "the " , "a ", "to " };
    NVStrings* tgts = NVStrings::create_from_array(htgts.data(),htgts.size());
    std::vector<const char*> hrpls{"_ "};
    NVStrings* rpls = NVStrings::create_from_array(hrpls.data(),hrpls.size());

    NVStrings* got = strs->replace(*tgts,*rpls);

    std::vector<const char*> expected{ "_ quick brown fox jumps over _ lazy dog",
                                "_ fat cat lays next _ _ other accénted cat",
                                 "_ slow moving turtlé cannot catch _ bird",
                                 "which can be composéd together _ form _ more complete",
                                 "thé result does not include _ value in _ sum in",
                                 "", "absent stop words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(rpls);
    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

TEST_F(TestReplace, ReplaceMultiRegex)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    std::vector<const char*> htgts{ "\\bthe\\b" , "\\ba\\b", "\\bto\\b" };
    std::vector<const char*> hrpls{ "", ".", "2" };
    NVStrings* rpls = NVStrings::create_from_array(hrpls.data(),hrpls.size());

    NVStrings* got = strs->replace_re(htgts,*rpls);

    std::vector<const char*> expected{ " quick brown fox jumps over  lazy dog",
                                " fat cat lays next 2  other accénted cat",
                                 ". slow moving turtlé cannot catch  bird",
                                 "which can be composéd together 2 form . more complete",
                                 "thé result does not include  value in  sum in",
                                 "", "absent stop words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(rpls);
    NVStrings::destroy(strs);
}

TEST_F(TestReplace, ReplaceTokens)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    std::vector<const char*> htgts{ "the" , "a", "to" };
    NVStrings* tgts = NVStrings::create_from_array(htgts.data(),htgts.size());
    std::vector<const char*> hrpls{ "", ".", "2" };
    NVStrings* rpls = NVStrings::create_from_array(hrpls.data(),hrpls.size());

    NVStrings* got = NVText::replace_tokens(*strs,*tgts,*rpls);

    std::vector<const char*> expected{ " quick brown fox jumps over  lazy dog",
                                " fat cat lays next 2  other accénted cat",
                                 ". slow moving turtlé cannot catch  bird",
                                 "which can be composéd together 2 form . more complete",
                                 "thé result does not include  value in  sum in",
                                 "", "absent stop words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(rpls);
    NVStrings::destroy(tgts);
    NVStrings::destroy(strs);
}

TEST_F(TestReplace, ReplaceBackrefs)
{
    NVStrings* strs = NVStrings::create_from_array(hstrs.data(),hstrs.size());

    NVStrings* got = strs->replace_with_backrefs("(\\w) (\\w)", "\\1-\\2");

    std::vector<const char*> expected{ "the-quick-brown-fox-jumps-over-the-lazy-dog",
                                 "the-fat-cat-lays-next-to-the-other-accénted-cat",
                                 "a-slow-moving-turtlé-cannot-catch-the-bird",
                                 "which-can-be-composéd-together-to-form-a more-complete",
                                 "thé-result-does-not-include-the-value-in-the-sum-in",
                                 "", "absent-stop-words" };

    EXPECT_TRUE( verify_strings(got,expected.data()) );

    NVStrings::destroy(got);
    NVStrings::destroy(strs);
}
