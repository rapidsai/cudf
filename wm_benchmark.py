# wm benchmark
import cudf
import glob
directory = "/home/coder/cudf/generated_data/WM_MOCKED_2/"
df = cudf.read_parquet([filename for filename in glob.iglob(f'{directory}/*.parquet')][:8])
STRING = cudf.dtype(str)
dtype = {
          "JEBEDJPKEFHPHGLLGPM": STRING,
          "FLMEPG": cudf.StructDtype({"CGEGPD":STRING}),
          "JACICCCIMMHJHKPDED": cudf.StructDtype({
            "OGGC":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"MDGA":STRING}) )})
          }),
          "AGHF": cudf.StructDtype({
            "DPKEAPDACLPHGPEMH":STRING,
            "ONNILHPABGIKKFJOEK":STRING,
            "FFFPOENCNBBNOOMOJGDBNIPD":STRING
          }),
          "AENBHHGIABBBDDGOEI": cudf.StructDtype({
            "PIGOFCPIPPBNNB":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "CCBJKBHGPBJCKFPCBHGLOAFE":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "LMPCGHBIJGCIPDPNELPBCOP":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "PKBGI":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "ILPIJKBLDB":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "GHBBEOAC":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "EKGPKGCJPMI":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "BDEGLFGMCPKOCNDGJMFPANNBPK":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "LILJMMPPO":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "EAGCHCMLMOLGJK":cudf.StructDtype({
              "BEACAHEBBO":cudf.StructDtype({
                "BNLFCI":STRING,
                "GPIHMJ":STRING
              }),
              "CGEGPD":cudf.ListDtype(cudf.StructDtype({
                "GJFKCFJELPJEDBAD":STRING,
                "GMFDD":STRING
              }))
            }),
            "PMJPCGCHAALKBPKHDM":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "OCFGAF":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "GMJICFMBNPLBEOLMGDN":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "CBMI":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "NPAGLLFCHAI":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "LFKAJEPMJPLGLICEEMAHFEJGPLGIAKPIOPPP":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "HGNHKIOEGKIJJJPEC":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "JAGGKPKOICKOBABAJPNHF":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "PLEJAKDBBGLCDLGDIBHPPBHB":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "MMNHNPKGLLBJMAOGOCBEOIOKIM":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "JLKDBLFFFPPCNANBKMELJKFOPKPNC":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "OCJGMOAJJKBKNCHOJKBJG":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "PMOAGIJAFOGGLINIOEBFGHBN":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "JPDILOFKPCNBKDB":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "CPBFNDGC":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "KPOPPCFLFCNAPIJEDJDGGFBOPLDCMLLGOMO":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "LBDGCNJNOGMJPNHMLLBMA":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "EIHBDLNJDOAHPMCNGGLLEF":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "GIPPDMMAFOBAALMHMGJBM":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "FKBODHACMMGHL":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({
              "KMEJHDA":STRING,
              "CJKIKCGA":STRING
            }))}),
            "HFFDKEDMFBAKEHHM":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "KGJLLAPHJNKCEOIAMCAABCJP":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "KLJNBPLECGCA":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "NBJNFKKKCHEGCABDGKG":cudf.StructDtype({
              "BEACAHEBBO":cudf.StructDtype({
                "BNLFCI":STRING,
                "GPIHMJ":STRING
              }),
              "CGEGPD":cudf.ListDtype(cudf.StructDtype({
                "GJFKCFJELPJEDBAD":STRING,
                "GMFDD":STRING
              }))
            }),
            "AOHKGCPAOGANLKEJDLMIGDD":cudf.StructDtype({"BEACAHEBBO":cudf.StructDtype({
              "BNLFCI":STRING,
              "GPIHMJ":STRING
            })}),
            "IKHLECMHMONKLKIBD":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "PNJPGEHPDLMPBDMFPLKABFFGG":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "IGAJPHHGOENI":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "LDPMFNAGLJGDMFOLAKH":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({
              "KMEJHDA":STRING,
              "CJKIKCGA":STRING
            }))}),
            "BFAJJIOLJBEOMFKLE":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))}),
            "DOONHL":cudf.StructDtype({"CGEGPD":cudf.ListDtype(cudf.StructDtype({"GMFDD":STRING}))})
          }),
          "OCIKAF": STRING
}


if df["columnC"].str.contains("\n").any()==True:
    #error
    print("Error")
    exit(1)

json_data = df["columnC"].str.cat(sep="\n", na_rep="{}")
#1.9812268866226077 GB
from io import StringIO
import time
import nvtx
with nvtx.annotate("from_json", color="purple"):
  start_time = time.time()
  df2 = cudf.read_json(StringIO(json_data), dtype=dtype, lines=True, prune_columns=True, on_bad_lines='recover')
  print("--- %s seconds ---" % (time.time() - start_time))
  print("Throughput: ", len(json_data)/(1024*1024*1024)/(time.time() - start_time), "GB/s")

