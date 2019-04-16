# pragma once


// DOXY
class JsonReader {
  json_read_arg* args_ = nullptr;
  std::vector<gdf_column_wrapper> columns_;
public:
  JsonReader(json_read_arg* args): args_(args){}

  void parse();
  void storeColumns(json_read_arg *out_args);


};