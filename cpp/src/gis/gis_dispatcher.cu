
#include "gis_functions.cuh"

gdf_error gdf_point_in_polygon(gdf_column* polygon_latitudes, gdf_column* polygon_longitudes, gdf_column* point_latitudes, gdf_column* point_longitudes,
gdf_column* output)
{
    //TODO: add type dispatcher to mix and match types in future
   return gdf_point_in_polygon_caller(polygon_latitudes, polygon_longitudes, point_latitudes, point_longitudes, output);
}