
#pragma once

/**  IO
 *
 *
 */

gdf_error read_csv(csv_read_arg *args);

gdf_error gdf_to_csr(gdf_column **gdfData, int num_cols, csr_gdf *csrReturn);
