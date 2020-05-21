#include <string>
#include <cudf/table/table_view.hpp>

class CudfWrapper {
  cudf::mutable_table_view mtv;

  public:
    // Creates a Wrapper around an existing cuDF Dataframe object
    CudfWrapper(cudf::mutable_table_view table_view);

    ~CudfWrapper();

    void tenth_mm_to_inches(int column_index);

    // Performs MM to Inches conversion on the GPU device
    void mm_to_inches(int column_index);

};
