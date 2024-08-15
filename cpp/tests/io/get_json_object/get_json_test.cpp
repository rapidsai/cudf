
#include "spark/get_json_object.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include "cudf/column/column_view.hpp"
#include "cudf/copying.hpp"
#include "cudf/types.hpp"
#include "cudf_test/debug_utilities.hpp"
#include <cudf/io/new_json_object.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#define wrapper cudf::test::fixed_width_column_wrapper
using float_wrapper        = wrapper<float>;
using float64_wrapper      = wrapper<double>;
using int_wrapper          = wrapper<int>;
using int8_wrapper         = wrapper<int8_t>;
using int16_wrapper        = wrapper<int16_t>;
using int64_wrapper        = wrapper<int64_t>;
using timestamp_ms_wrapper = wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>;
using bool_wrapper         = wrapper<bool>;

using cudf::data_type;
using cudf::type_id;
using cudf::type_to_id;
using cudf::spark_rapids_jni::json_path_t;
using cudf::spark_rapids_jni::path_instruction_type;

void print_json_path(json_path_t const& paths) {
   for (const auto& component : paths) {
            path_instruction_type type = std::get<0>(component);
            std::string name = std::get<1>(component);
            int32_t index = std::get<2>(component);

            std::cout << "(";
            switch (type) {
                case path_instruction_type::WILDCARD: std::cout << "WILDCARD"; break;
                case path_instruction_type::INDEX: std::cout << "INDEX"; break;
                case path_instruction_type::NAMED: std::cout << "NAMED"; break;
            }
            std::cout << ", \"" << name << "\", " << index << ")\n";
        }
}

std::vector<json_path_t> pathstrs_to_json_paths(std::vector<std::string> const& paths) {
  std::vector<json_path_t> json_paths;
  const std::string delims = ".[";
  json_paths.reserve(paths.size());
  for(std::string_view strpath : paths) {
    size_t start = 0;
    json_path_t jpath;
    while (start < strpath.size()) {
          size_t end = strpath.find_first_of(delims, start);
          std::string_view this_path;
          if (end == std::string_view::npos) {
            this_path = strpath.substr(start);
          } else {
            this_path = strpath.substr(start, end - start);
            start = end + 1;
          }
          if (this_path == "$") continue;
          else if (this_path == "*]") {
            jpath.emplace_back(path_instruction_type::WILDCARD, "", -1);
          } else if (this_path.back()==']') {
            auto index = std::stoi(std::string(this_path.substr(0, this_path.size()-1)));
            jpath.emplace_back(path_instruction_type::INDEX, "", index);
          } else {
            jpath.emplace_back(path_instruction_type::NAMED, this_path, -1);
          }
          if (end == std::string_view::npos) break;
      }
      json_paths.push_back(jpath);
  }
  return json_paths;
}
auto get_json_object_single(cudf::column_view input, std::string path, std::string lineno) {
  // get_json_object(columnA, '$.EIFGPHGLOPELFBN')
  cudf::scoped_range rng(lineno);
  auto json_paths = pathstrs_to_json_paths({path});
  print_json_path(json_paths[0]);
  return std::move(cudf::spark_rapids_jni::get_json_object_multiple_paths2(input, json_paths, cudf::get_default_stream())[0]);
}

int count_leaves(cudf::column_view const& col) {
  if (col.type().id()==type_id::STRING) return 1;
  // if (col.type().id()!=type_id::LIST and col.type().id()!=type_id::STRUCT) return 1;
  int count=0;
  for(auto chit = col.child_begin(); chit != col.child_end(); chit++) {
    count += count_leaves(*chit);
  }
  return count;
}

auto get_json_object_multiple2(cudf::column_view input, std::vector<std::string>const& paths, std::string lineno) {
  // get_json_object(columnA, '$.EIFGPHGLOPELFBN')
  cudf::scoped_range rng("m"+lineno);
  auto json_paths = pathstrs_to_json_paths(paths);
  // print_json_path(json_paths[0]);
  auto result = cudf::spark_rapids_jni::get_json_object_multiple_paths2(input, json_paths, cudf::get_default_stream());
  // TODO check if returned leaf columns are all STRING and count is equal.
  auto num_outputs = paths.size();
  std::cout<<"num_outputs:"<<num_outputs<<"\n";
  auto num_leaf= 0;
  for(auto& ch : result)
  if(ch)
    num_leaf += count_leaves(*ch);
  std::cout<<"num_leaf:"<<num_leaf<<"\n";
  return result;
}


#define get_json_object(a, b) get_json_object_single(a, b, std::to_string(__LINE__))
#define get_json_object_multiple(...) get_json_object_multiple2(__VA_ARGS__, std::to_string(__LINE__))

auto get_json_object_multiple1(cudf::column_view input, std::vector<std::string>const& paths, std::string lineno) {
  cudf::scoped_range rng("s"+lineno);
  auto json_paths = pathstrs_to_json_paths(paths);
  return ::spark_rapids_jni::get_json_object_multiple_paths(input, json_paths, size_t(4)<<30, 2, cudf::get_default_stream());
}

#define old_get_json_object_multiple1(...) get_json_object_multiple1(__VA_ARGS__, std::to_string(__LINE__))

/**
 * @brief Base test fixture for JSON reader tests
 */
struct GetJsonTest : public cudf::test::BaseFixture {};

std::vector<std::string> files{
"part-00000-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00004-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00008-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00010-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00013-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00015-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00018-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00021-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00022-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
// "part-00030-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet",
};

TEST_F(GetJsonTest, DifferentGetJSONObject)
{

  cudf::scoped_range at{"DifferentGetJSONObject"};
  const std::string filename = "part-00000-505e98e9-a5c8-4720-8bb4-d6cc96625744-c000.snappy.parquet";
  const std::string dir = "/home/coder/cudf/generated_data/WM_MOCKED_2/";
  const std::string filepath = dir + "/" + filename;
  std::vector<std::string> full_files;
  std::transform(files.begin(), files.end(), std::back_inserter(full_files), [dir](auto f) {  return dir + f;});
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{full_files})
          .columns({"columnA", "columnB", "columnC", "columnD", "columnE"});
  auto result = cudf::io::read_parquet(read_opts);

  // std::string requested_column = "columnA";
  // auto iter = std::find_if(result.metadata.schema_info.begin(), result.metadata.schema_info.end(), [requested_column](auto col_info) {
  //   return col_info.name == requested_column;
  // });
  // if(iter==result.metadata.schema_info.end()) throw std::out_of_range("Can't find expected column " + requested_column);
  // auto col_index = iter - result.metadata.schema_info.begin();
  // auto input = result.tbl->get_column(col_index);
  auto columnA = result.tbl->get_column(0).view();
  auto columnB = result.tbl->get_column(1).view();
  auto columnC = result.tbl->get_column(2).view();
  auto columnD = result.tbl->get_column(3).view();
  auto columnE = result.tbl->get_column(4).view();

  for(auto c: result.metadata.schema_info) {
    std::cout<< c.name <<",";
  } std::cout<< "\n";

//  {
//   cudf::scoped_range at{"SINGLE_CALL", nvtx3::rgb{116,0,255}};
//   #include "all_objects.cpp"
//  }
  // {
  //   cudf::scoped_range at{"MULTI_CALL", nvtx3::rgb{210, 22, 210}};
  //   #include "multi_objects.cpp"
  // }
  // {
  //   cudf::scoped_range at{"SPARK_MULTI_CALL", nvtx3::rgb{210, 22, 210}};
  //   #include "old_multi_objects.cpp"
  // }

  {
    auto res1 = old_get_json_object_multiple1(columnA,
                                              {
                                                "$.EIFGPHGLOPELFBN",
                                              });
    auto res2 = get_json_object_multiple(columnA,
                                         {
                                           "$.EIFGPHGLOPELFBN",
                                         });
    EXPECT_EQ(res1.size(), res2.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(res1[0]->view(), res2[0]->view());
  }
  //  {
  //   auto res1 = old_get_json_object_multiple1(columnB,
  //   {"$[0].GHPKNICLNDAGCNDBMFGEK[0].KIKNFPPAPGDO.KLFALIBALPPK.HGABIFNPHAHHGP",}); 
  //   auto res2 = get_json_object_multiple(columnB,
  //   {"$[0].GHPKNICLNDAGCNDBMFGEK[0].KIKNFPPAPGDO.KLFALIBALPPK.HGABIFNPHAHHGP",});
  //   EXPECT_EQ(res1.size(), res2.size());
  //   CUDF_TEST_EXPECT_COLUMNS_EQUAL(res1[0]->view(), res2[0]->view());
  //  }

{
  std::vector<std::string> paths = {
"$.JEBEDJPKEFHPHGLLGPM",
"$.FLMEPG.CGEGPD",
// "$.JACICCCIMMHJHKPDED.ACHCPIHLFCPHMBPNKJNOLNO.CGEGPD[*].GMFDD", // FIXME: wildcard
"$.JACICCCIMMHJHKPDED.OGGC.CGEGPD[0].MDGA",
"$.AGHF.DPKEAPDACLPHGPEMH",
"$.AGHF.ONNILHPABGIKKFJOEK",
"$.AGHF.FFFPOENCNBBNOOMOJGDBNIPD",
// "$.AENBHHGIABBBDDGOEI.POFNDBFHDEJ.CGEGPD[*].GMFDD", // FIXME: wildcard
"$.AENBHHGIABBBDDGOEI.PIGOFCPIPPBNNB.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.CCBJKBHGPBJCKFPCBHGLOAFE.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.LMPCGHBIJGCIPDPNELPBCOP.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.PKBGI.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.ILPIJKBLDB.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.GHBBEOAC.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.EKGPKGCJPMI.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.BDEGLFGMCPKOCNDGJMFPANNBPK.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.LILJMMPPO.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.BEACAHEBBO.BNLFCI",
"$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.BEACAHEBBO.GPIHMJ",
"$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.CGEGPD[0].GJFKCFJELPJEDBAD",
"$.AENBHHGIABBBDDGOEI.EAGCHCMLMOLGJK.CGEGPD[0].GMFDD",
// "$.AENBHHGIABBBDDGOEI.DLJPDEPFEKDCKBI.CGEGPD[*].GMFDD", // FIXME: wildcard
"$.AENBHHGIABBBDDGOEI.PMJPCGCHAALKBPKHDM.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.OCFGAF.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.GMJICFMBNPLBEOLMGDN.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.CBMI.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.NPAGLLFCHAI.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.LFKAJEPMJPLGLICEEMAHFEJGPLGIAKPIOPPP.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.HGNHKIOEGKIJJJPEC.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.JAGGKPKOICKOBABAJPNHF.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.PLEJAKDBBGLCDLGDIBHPPBHB.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.MMNHNPKGLLBJMAOGOCBEOIOKIM.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.JLKDBLFFFPPCNANBKMELJKFOPKPNC.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.OCJGMOAJJKBKNCHOJKBJG.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.PMOAGIJAFOGGLINIOEBFGHBN.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.JPDILOFKPCNBKDB.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.CPBFNDGC.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.KPOPPCFLFCNAPIJEDJDGGFBOPLDCMLLGOMO.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.LBDGCNJNOGMJPNHMLLBMA.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.EIHBDLNJDOAHPMCNGGLLEF.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.GIPPDMMAFOBAALMHMGJBM.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.FKBODHACMMGHL.CGEGPD[0].KMEJHDA",
"$.AENBHHGIABBBDDGOEI.FKBODHACMMGHL.CGEGPD[0].CJKIKCGA",
"$.AENBHHGIABBBDDGOEI.HFFDKEDMFBAKEHHM.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.KGJLLAPHJNKCEOIAMCAABCJP.CGEGPD[1].GMFDD",
"$.AENBHHGIABBBDDGOEI.KLJNBPLECGCA.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.NBJNFKKKCHEGCABDGKG.BEACAHEBBO.BNLFCI",
"$.AENBHHGIABBBDDGOEI.NBJNFKKKCHEGCABDGKG.BEACAHEBBO.GPIHMJ",
"$.AENBHHGIABBBDDGOEI.NBJNFKKKCHEGCABDGKG.CGEGPD[0].GJFKCFJELPJEDBAD",
"$.AENBHHGIABBBDDGOEI.NBJNFKKKCHEGCABDGKG.CGEGPD[0].GMFDD",
// "$.AENBHHGIABBBDDGOEI.AOHKGCPAOGANLKEJDLMIGDD.BEACAHEBBO.BNLFCI", // FIXME mixed type as struct
// "$.AENBHHGIABBBDDGOEI.AOHKGCPAOGANLKEJDLMIGDD.BEACAHEBBO.GPIHMJ", // FIXME as struct
// "$.AENBHHGIABBBDDGOEI.AOHKGCPAOGANLKEJDLMIGDD[0].GMFDD",          // FIXME as list
// "$.AENBHHGIABBBDDGOEI.IKHLECMHMONKLKIBD.CGEGPD[0].GMFDD",         // FIXME as list
"$.AENBHHGIABBBDDGOEI.PNJPGEHPDLMPBDMFPLKABFFGG.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.IGAJPHHGOENI.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.LDPMFNAGLJGDMFOLAKH.CGEGPD[0].KMEJHDA",
"$.AENBHHGIABBBDDGOEI.LDPMFNAGLJGDMFOLAKH.CGEGPD[0].CJKIKCGA",
"$.AENBHHGIABBBDDGOEI.BFAJJIOLJBEOMFKLE.CGEGPD[0].GMFDD",
"$.AENBHHGIABBBDDGOEI.DOONHL.CGEGPD[0].GMFDD",
"$.OCIKAF",
// "$.IBMBCGNOCGCPCEN[*].GLNLBEA", // FIXME: wildcard
};

    auto columnCC = cudf::slice(columnC, {0, 20})[0];
    auto res1 = old_get_json_object_multiple1(columnCC, paths);
    auto res2 = get_json_object_multiple(columnCC, paths);
     EXPECT_EQ(res1.size(), res2.size());
    for(size_t i=0; i<res1.size(); i++) {
      std::cout<<i<<"\n";
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(res1[i]->view(), res2[i]->view());
    }
    // cudf::test::print(res1[54]->view());
    // cudf::test::print(res2[54]->view());
  }
  {
    auto res1 = old_get_json_object_multiple1(columnD, {"$.KPIGLEDEOCFELKLJLAFE",});
    auto res2 = get_json_object_multiple(columnD, {"$.KPIGLEDEOCFELKLJLAFE",});
    EXPECT_EQ(res1.size(), res2.size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(res1[0]->view(), res2[0]->view());
  }
  {
    std::cout<<"DEBUG:\n";
    auto columnS = columnE;// = cudf::slice(columnE, {0, 2})[0];
    auto res1 = old_get_json_object_multiple1(columnS, {
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].GPGACKDIBMPAKJMDMJ",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].NOIIFOJOPJP",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].CEJOOHNF",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].HODJK[0].HHKEKMIIGI",
// "$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].KDGJICMEANMA[*].ILEADAN", //FIXME: wildcard null count mismatch
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].OKPLFLHHEBDJELFA",
// "$.NHKDIEPJNND.CHNFGBB.KIKNFPPAPGDO.KLFALIBALPPK.HGABIFNPHAHHGP", //FIXME: float
"$.NHKDIEPJNND.IHIIKIHHMPFL.KCCCHAM.KCCCHAM",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].CEJOOHNF",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].HODJK[0].HHKEKMIIGI",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].OKPLFLHHEBDJELFA",
"$.JJKPNPFMNICGLC.GGLF.JKKJDAKAB",
"$.KPIGLEDEOCFELKLJLAFE",
"$.PACKGGMDGCLEHD.IAFMNJMMNJPDAAHND",
"$.PACKGGMDGCLEHD.MNIMBEMMOJFHILDMDBML",
});
    auto res2 = get_json_object_multiple(columnS, {
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].GPGACKDIBMPAKJMDMJ",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].NOIIFOJOPJP",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].CEJOOHNF",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].HODJK[0].HHKEKMIIGI",
// "$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].KDGJICMEANMA[*].ILEADAN",
"$.NHKDIEPJNND.DPBFKLKAKDHLMDLIONCCLJ[0].OKPLFLHHEBDJELFA",
// "$.NHKDIEPJNND.CHNFGBB.KIKNFPPAPGDO.KLFALIBALPPK.HGABIFNPHAHHGP",
"$.NHKDIEPJNND.IHIIKIHHMPFL.KCCCHAM.KCCCHAM",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].CEJOOHNF",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].HODJK[0].HHKEKMIIGI",
"$.KFPJHMGFEELFG[0].AFHKGOFNFID[0].DPBFKLKAKDHLMDLIONCCLJ[0].OKPLFLHHEBDJELFA",
"$.JJKPNPFMNICGLC.GGLF.JKKJDAKAB",
"$.KPIGLEDEOCFELKLJLAFE",
"$.PACKGGMDGCLEHD.IAFMNJMMNJPDAAHND",
"$.PACKGGMDGCLEHD.MNIMBEMMOJFHILDMDBML",
});
    EXPECT_EQ(res1.size(), res2.size());
    for(size_t i=0; i<res1.size(); i++) {
      std::cout<<i<<"\n";
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(res1[i]->view(), res2[i]->view());
    }

    // cudf::test::print(res1[0]->view());
    // cudf::test::print(res2[0]->view());
   }
}